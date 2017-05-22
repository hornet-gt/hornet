/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date May, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "Core/cuStinger.hpp"
#include "Core/BatchUpdateKernels.cuh"
#include "Support/Device/CubWrapper.cuh"
#include "Support/Device/PrintExt.cuh"


namespace custinger {

void cuStinger::insertEdgeBatch(BatchUpdate& batch_update) noexcept {
    insertEdgeBatch(batch_update, [] __device__ (const Edge& a, const Edge& b) {
                                      return a.dst() == b.dst();
                                  });
}

template<typename EqualOp>
void cuStinger::insertEdgeBatch(BatchUpdate& batch_update,
                                const EqualOp& equal_op) noexcept {
    const unsigned BLOCK_SIZE = 256;
    size_t batch_size = batch_update.size();

    vid_t     *d_batch_src_sorted, *d_batch_dst_sorted, *d_unique;
    bool      *d_flags;
    int       *d_counts, *d_queue_size;
    degree_t  *d_degree;
    UpdateStr *d_queue;
    edge_t    **d_old_ptrs, **d_new_ptrs;
    //--------------------------------------------------------------------------
    ////////////////
    // ALLOCATION //
    ////////////////
    vid_t* d_batch_src = batch_update.src_ptr();
    vid_t* d_batch_dst = batch_update.dst_ptr();
    cuMalloc(d_batch_src_sorted, batch_size);
    cuMalloc(d_batch_dst_sorted, batch_size);
    cuMalloc(d_flags, batch_size);
    cuMalloc(d_counts, batch_size + 1);
    cuMalloc(d_unique, batch_size);
    cuMalloc(d_queue, batch_size);      // allocate |queue| = num. unique src
    cuMalloc(d_queue_size, 1);
    cuMalloc(d_degree, batch_size + 1);
    cuMemset0x00(d_queue_size);
    cuMemset0x00(d_degree + batch_size);
    cuMemset0x00(d_counts + batch_size);

    auto h_old_ptrs = new edge_t*[batch_size];
    auto h_new_ptrs = new edge_t*[batch_size];
    auto    h_queue = new UpdateStr[batch_size];
    cuMalloc(d_old_ptrs, batch_size);
    cuMalloc(d_new_ptrs, batch_size);
    //cudaMallocHost()cudaFreeHost
    vid_t* tmp1, *tmp2;
    cuMalloc(tmp1, batch_size);
    cuMalloc(tmp2, batch_size);

    std::cout << d_batch_src << " " << d_batch_dst << std::endl;
    std::cout << ((uint64_t)d_batch_src % 16) << " " << ((uint64_t)d_batch_dst % 16) << std::endl;

    //--------------------------------------------------------------------------
    //////////////
    // CUB INIT //
    //////////////
    //*xlib::CubSortByKey<vid_t, vid_t> sort_cub(d_batch_src, batch_size,
    //                                            d_batch_src_sorted, _nV);
    /*xlib::CubSortPairs2<vid_t, vid_t> sort_cub(d_batch_src, d_batch_dst,
                                               batch_size, _nV);

    xlib::PartitionFlagged<vid_t> partition_cub1(d_batch_src, d_flags,
                                                 batch_size, d_batch_src);

    xlib::PartitionFlagged<vid_t> partition_cub2(d_batch_dst, d_flags,
                                                 batch_size, d_batch_dst);*/
    xlib::CubSortPairs2<vid_t, vid_t> sort_cub(d_batch_src, d_batch_dst,
                                               batch_size, d_batch_src_sorted,
                                               d_batch_dst_sorted, _nV);

    xlib::PartitionFlagged<vid_t> partition_cub1(d_batch_src_sorted, d_flags,
                                                 batch_size, tmp1/*d_batch_src*/);

    xlib::PartitionFlagged<vid_t> partition_cub2(d_batch_dst_sorted, d_flags,
                                                 batch_size, d_batch_dst);

    xlib::CubRunLengthEncode<vid_t> runlength_cub(d_batch_src, batch_size,
                                                  d_unique, d_counts);

    //==========================================================================
    sort_cub.run();                        // batch sorting

    cu::printArray(d_batch_src_sorted, batch_size);
    cu::printArray(d_batch_dst_sorted, batch_size);

    findDuplicateKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(batch_update.size()), BLOCK_SIZE >>>
        (device_data(), batch_update, equal_op, d_flags);
    CHECK_CUDA_ERROR

    std::cout << "findDuplicateKernel " << std::endl;
    cu::printArray(d_flags, batch_size);

    partition_cub1.run();                   //remove dublicate from batch_src

    std::cout << "partition_cub1 " << std::endl;

    partition_cub2.run();                   //remove dublicate from batch_dst

    std::cout << "partition_cub2 " << std::endl;
    int   num_items = runlength_cub.run();  //find unique src and occurences
    auto vertex_ptr = reinterpret_cast<VertexBasicData*>(_d_vertex_ptrs[0]);

    std::cout << "num_items " << num_items << std::endl;

    findSpaceRequest
        <<< xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE >>>
        (vertex_ptr, d_unique, d_counts, num_items, d_queue, d_queue_size);
    CHECK_CUDA_ERROR

    int h_queue_size;
    cuMemcpyToHostAsync(d_queue_size, h_queue_size);

    //==========================================================================
    ///////////////////////
    // MEMORY ALLOCATION //
    ///////////////////////
    if (h_queue_size > 0) {
        collectInfoForHost
            <<< xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE >>>
            (vertex_ptr, d_queue, h_queue_size, d_degree, d_old_ptrs);
        CHECK_CUDA_ERROR

        cuMemcpyToHostAsync(d_old_ptrs, h_queue_size, h_old_ptrs);
        cuMemcpyToHostAsync(d_queue,    h_queue_size, h_queue);

        for (int i = 0; i < h_queue_size; i++)
            mem_manager.remove(h_old_ptrs[i], h_queue[i].old_degree);
        for (int i = 0; i < h_queue_size; i++)
            h_new_ptrs[i] = mem_manager.insert(h_queue[i].new_degree).second;

        cuMemcpyToDeviceAsync(h_new_ptrs, h_queue_size, d_new_ptrs);

        //----------------------------------------------------------------------
        updateVertexBasicData
            <<< xlib::ceil_div<BLOCK_SIZE>(h_queue_size), BLOCK_SIZE >>>                        // update data structures
            (vertex_ptr, d_queue, h_queue_size, d_new_ptrs);
        CHECK_CUDA_ERROR

        //----------------------------------------------------------------------
        // WORKLOAD COMPUTATION
        xlib::CubExclusiveSum<degree_t> prefixsum1(d_degree, h_queue_size + 1);
        xlib::CubExclusiveSum<int>      prefixsum2(d_counts, num_items + 1);
        prefixsum1.run();

        const auto& d_prefixsum = d_degree;     //alias
        int      prefixsum_size = h_queue_size; //alias

        degree_t prefixsum_total;               //get the total
        cuMemcpyToHostAsync(d_prefixsum + prefixsum_size, prefixsum_total);

        //----------------------------------------------------------------------
        //copy adjcency lists to new memory locations
        const int     SMEM = xlib::SMemPerBlock<BLOCK_SIZE, degree_t>::value;
        int partition_size = xlib::ceil_div<SMEM>(prefixsum_total);

        bulkCopyAdjLists<BLOCK_SIZE, SMEM> <<< partition_size, BLOCK_SIZE >>>
            (d_prefixsum, prefixsum_size, d_old_ptrs, d_new_ptrs);
        //----------------------------------------------------------------------
        //Merge sort
        prefixsum2.run();

        //bulkCopyAdjLists <<< partition_size, BLOCK_SIZE >>>
        //    (d_prefixsum, d_old_ptrs, d_new_ptrs);

        mergeAdjListKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE >>>
            (device_data(), batch_update, num_items, d_unique, d_counts);
    }
}

} // namespace custinger
