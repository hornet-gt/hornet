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
#include "Core/BatchInsertKernels.cuh"
#include "Core/BatchDeleteKernels.cuh"
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
    int       *d_counts, *d_num_realloc;
    degree_t  *d_degree, *d_degree_changed;
    UpdateStr *d_queue_realloc;
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
    cuMalloc(d_degree_changed, batch_size);
    cuMalloc(d_queue_realloc, batch_size);      // allocate |queue| = num. unique src
    cuMalloc(d_num_realloc, 1);
    cuMalloc(d_degree, batch_size + 1);
    cuMemset0x00(d_num_realloc);
    cuMemset0x00(d_degree + batch_size);
    cuMemset0x00(d_counts + batch_size);

    auto      h_old_ptrs = new edge_t*[batch_size];
    auto      h_new_ptrs = new edge_t*[batch_size];
    auto h_queue_realloc = new UpdateStr[batch_size];
    cuMalloc(d_old_ptrs, batch_size);
    cuMalloc(d_new_ptrs, batch_size);
    //cudaMallocHost()cudaFreeHost
    //--------------------------------------------------------------------------
    //////////////
    // CUB INIT //
    //////////////
    //*xlib::CubSortByKey<vid_t, vid_t> sort_cub(d_batch_src, batch_size,
    //                                            d_batch_src_sorted, _nV);
    xlib::CubSortPairs2<vid_t, vid_t> sort_cub(d_batch_src, d_batch_dst,
                                               batch_size, _nV);

    xlib::PartitionFlagged<vid_t> partition_cub1(d_batch_src, d_flags,
                                                 batch_size, d_batch_src);

    xlib::PartitionFlagged<vid_t> partition_cub2(d_batch_dst, d_flags,
                                                 batch_size, d_batch_dst);
    //==========================================================================
    sort_cub.run();                        // batch sorting
    //cu::printArray(d_batch_src, batch_size);
    //cu::printArray(d_batch_dst, batch_size);

    findDuplicateKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(batch_update.size()), BLOCK_SIZE >>>
        (device_data(), batch_update, equal_op, d_flags);
    CHECK_CUDA_ERROR

    int num_noduplicate = partition_cub1.run();//remove dublicate from batch_src
    partition_cub2.run();                      //remove dublicate from batch_dst

    xlib::CubRunLengthEncode<vid_t> runlength_cub(d_batch_src, num_noduplicate,
                                                  d_unique, d_counts);
    int num_uniques = runlength_cub.run();     //find unique src and occurences
    auto vertex_ptr = reinterpret_cast<VertexBasicData*>(_d_vertex_ptrs[0]);
    //std::cout <<   "      nodup: " << num_noduplicate
    //          << "\nnum_uniques: " << num_uniques << std::endl;
    //cu::printArray(d_unique, num_uniques);
    //cu::printArray(d_counts, num_uniques);

    findSpaceRequest
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (vertex_ptr, d_unique, d_counts, num_uniques,
         d_queue_realloc, d_num_realloc, d_degree_changed);
    CHECK_CUDA_ERROR

    int h_num_realloc;
    cuMemcpyToHostAsync(d_num_realloc, h_num_realloc);

    //std::cout << "h_num_realloc: " << h_num_realloc <<std::endl;
    //==========================================================================
    ///////////////////////
    // MEMORY ALLOCATION //
    ///////////////////////
    if (h_num_realloc > 0) {
        collectInfoForHost
            <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
            (vertex_ptr, d_queue_realloc, h_num_realloc, d_degree, d_old_ptrs);
        CHECK_CUDA_ERROR

        cuMemcpyToHostAsync(d_old_ptrs, h_num_realloc, h_old_ptrs);
        cuMemcpyToHostAsync(d_queue_realloc, h_num_realloc, h_queue_realloc);
        //xlib::printArray(h_queue_realloc, h_num_realloc);

        for (int i = 0; i < h_num_realloc; i++) {
            auto new_degree = h_queue_realloc[i].new_degree;
            h_new_ptrs[i]   = mem_manager.insert(new_degree).second;
        }
        for (int i = 0; i < h_num_realloc; i++)
            mem_manager.remove(h_old_ptrs[i], h_queue_realloc[i].old_degree);

        cuMemcpyToDeviceAsync(h_new_ptrs, h_num_realloc, d_new_ptrs);
        //----------------------------------------------------------------------
        updateVertexBasicData
            <<< xlib::ceil_div<BLOCK_SIZE>(h_num_realloc), BLOCK_SIZE >>>                        // update data structures
            (vertex_ptr, d_queue_realloc, h_num_realloc, d_new_ptrs);
        CHECK_CUDA_ERROR

        //----------------------------------------------------------------------
        // WORKLOAD COMPUTATION
        xlib::CubExclusiveSum<degree_t> prefixsum1(d_degree, h_num_realloc + 1);
        xlib::CubExclusiveSum<int>      prefixsum2(d_counts, num_uniques + 1);
        prefixsum1.run();

        const auto& d_prefixsum = d_degree;      //alias
        int      prefixsum_size = h_num_realloc; //alias

        degree_t prefixsum_total;                //get the total
        cuMemcpyToHostAsync(d_prefixsum + prefixsum_size, prefixsum_total);
        //cu::printArray(d_prefixsum, prefixsum_size + 1, "prefix degree\n");
        //----------------------------------------------------------------------
        if (prefixsum_total > 0) {
            //copy adjcency lists to new memory locations
            const int SMEM = xlib::SMemPerBlock<BLOCK_SIZE, degree_t>::value;
            int num_blocks = xlib::ceil_div<SMEM>(prefixsum_total);

            bulkCopyAdjLists<BLOCK_SIZE, SMEM> <<< num_blocks, BLOCK_SIZE >>>
                (d_prefixsum, prefixsum_size + 1, d_old_ptrs, d_new_ptrs);
            CHECK_CUDA_ERROR
        }
        //----------------------------------------------------------------------
        //Merge sort
        prefixsum2.run();
        //cu::printArray(d_counts, num_uniques + 1, "prefix count\n");
        //bulkCopyAdjLists <<< num_noduplicate, BLOCK_SIZE >>>
        //    (d_prefixsum, d_old_ptrs, d_new_ptrs);
        //cu::printArray(d_batch_dst, num_noduplicate, "dst:\n");

        mergeAdjListKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
            (device_data(), d_degree_changed, batch_update,
             d_unique, d_counts, num_uniques);
        CHECK_CUDA_ERROR
    }
}

//==============================================================================
//==============================================================================

void cuStinger::edgeDeletionsSorted(BatchUpdate& batch_update) noexcept {
    const unsigned BLOCK_SIZE = 256;
    size_t batch_size = batch_update.size();

    vid_t     *d_batch_src_sorted, *d_batch_dst_sorted, *d_unique;
    int       *d_counts;
    degree_t  *d_degree_old, *d_degree_new;
    byte_t    **d_ptrs_array;
    TmpData   *d_tmp;

    //--------------------------------------------------------------------------
    ////////////////
    // ALLOCATION //
    ////////////////
    vid_t* d_batch_src = batch_update.src_ptr();
    vid_t* d_batch_dst = batch_update.dst_ptr();
    cuMalloc(d_batch_src_sorted, batch_size);
    cuMalloc(d_batch_dst_sorted, batch_size);
    cuMalloc(d_counts, batch_size);
    cuMalloc(d_unique, batch_size);
    cuMalloc(d_degree_old, batch_size + 1);
    cuMalloc(d_degree_new, batch_size + 1);
    cuMalloc(d_tmp, _nE);
    cuMalloc(d_ptrs_array, batch_size);
    //--------------------------------------------------------------------------
    //////////////
    // CUB INIT //
    //////////////
    xlib::CubSortPairs2<vid_t, vid_t> sort_cub(d_batch_src, d_batch_dst,
                                               batch_size, _nV);
    xlib::CubRunLengthEncode<vid_t> runlength_cub(d_batch_src, batch_size,
                                                  d_unique, d_counts);
    //==========================================================================
    sort_cub.run();                        // batch sorting
    cu::printArray(d_batch_src, batch_size);
    cu::printArray(d_batch_dst, batch_size);

    int num_uniques = runlength_cub.run();

    std::cout << num_uniques << std::endl;
    cu::printArray(d_counts, num_uniques, "d_counts\n");

    deleteEdgesKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
        (device_data(), batch_update);
    CHECK_CUDA_ERROR

    collectDataKernel   //modify also the vertices degree
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (device_data(), d_unique, d_counts, num_uniques,
         d_degree_old, d_degree_new, d_ptrs_array);

    cu::printArray(d_degree_old, num_uniques, "d_degree_old\n");
    cu::printArray(d_degree_new, num_uniques, "d_degree_new\n");

    xlib::CubExclusiveSum<degree_t> prefixsum1(d_degree_old, num_uniques + 1);
    xlib::CubExclusiveSum<degree_t> prefixsum2(d_degree_new, num_uniques + 1);
    prefixsum1.run();
    prefixsum2.run();

    cu::printArray(d_degree_old, num_uniques + 1, "d_degree_old\n");
    cu::printArray(d_degree_new, num_uniques + 1, "d_degree_new\n");

    degree_t total_degree_old, total_degree_new;                //get the total
    cuMemcpyToHostAsync(d_degree_old + num_uniques, total_degree_old);
    cuMemcpyToHostAsync(d_degree_new + num_uniques, total_degree_new);

    const int SMEM = xlib::SMemPerBlock<BLOCK_SIZE, degree_t>::value;

    moveDataKernel1<BLOCK_SIZE, SMEM>
        <<< xlib::ceil_div<BLOCK_SIZE>(total_degree_old), BLOCK_SIZE >>>
        (d_degree_old, num_uniques + 1, d_ptrs_array, d_tmp);

    cu::printArray(d_tmp, total_degree_old, "d_tmp old\n");

    xlib::CubSelect<degree_t> select(d_tmp, total_degree_old);
    int tmp_size_new = select.run_diff(-1);
    assert(total_degree_new == tmp_size_new);

    std::cout << total_degree_new << " , " << tmp_size_new << "\n";
    cu::printArray(d_tmp, total_degree_new, "d_tmp new\n");

    moveDataKernel2<BLOCK_SIZE, SMEM>
        <<< xlib::ceil_div<BLOCK_SIZE>(total_degree_new), BLOCK_SIZE >>>
        (d_degree_new, num_uniques + 1, d_tmp, d_ptrs_array);
}

} // namespace custinger
