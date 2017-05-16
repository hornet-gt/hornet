/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
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
#include "Support/Device/SafeCudaAPI.cuh"

namespace custinger {

inline BatchInit::BatchInit(int batch_size, const vid_t* src_array,
                            const vid_t* dst_array) noexcept :
                                _batch_size(batch_size) {
    _edge_ptrs[0] = reinterpret_cast<const byte_t*>(src_array);
    _edge_ptrs[1] = reinterpret_cast<const byte_t*>(dst_array);
}

template<typename... TArgs>
void BatchInit::insertEdgeData(TArgs... edge_data) noexcept {
    bind<2>(_edge_ptrs, edge_data...);
}

inline int BatchInit::size() const noexcept {
    return _batch_size;
}

inline const byte_t* BatchInit::edge_ptrs(int index) const noexcept {
    return _edge_ptrs[index];
}

//==============================================================================

inline BatchUpdate::BatchUpdate(BatchInit batch_init) noexcept :
                            _batch_size(batch_init.size()),
                            _batch_pitch(xlib::upper_approx<512>(_batch_size)) {
    byte_t* ptr;
    cuMalloc(ptr, _batch_pitch * (sizeof(edge_t) + sizeof(vid_t)));
    for (int i = 0; i < NUM_ETYPES + 1; i++) {
        _d_edge_ptrs[i] = ptr;
        cuMemcpyToDeviceAsync(batch_init.edge_ptrs(i), _batch_size,
                              _d_edge_ptrs[i]);
        ptr += _batch_pitch;
    }
}

inline BatchUpdate::~BatchUpdate() noexcept {
    cuFree(_d_edge_ptrs[0]);
}

#if defined(__NVCC__)

__host__ __device__ __forceinline__
int BatchUpdate::size() const noexcept {
    return _batch_size;
}

__device__ __forceinline__
vid_t BatchUpdate::src(int index) const noexcept {
    return reinterpret_cast<vid_t*>(_d_edge_ptrs[0])[index];
}

__device__ __forceinline__
vid_t BatchUpdate::dst(int index) const noexcept {
    return reinterpret_cast<vid_t*>(_d_edge_ptrs[1])[index];
}

__device__ __forceinline__
Edge BatchUpdate::edge(int index) const noexcept {
    return Edge(_d_edge_ptrs[1], index, _batch_pitch);
}

template<int INDEX>
__device__ __forceinline__
typename std::tuple_element<INDEX, VertexTypes>::type
BatchUpdate::field(int index) const noexcept {

}

#endif

//==============================================================================
/*
    const unsigned BLOCK_SIZE = 256;

    vid_t *d_batch_src, *d_batch_dest, *d_batch_src_sorted,
          *d_batch_dest_sorted, *d_unique;
    int   *d_counts, *d_queue_size;
    degree_t*  d_degree;
    UpdateStr* d_queue;
    edge_t**   d_old_ptrs, **d_new_ptrs;
    auto h_old_ptrs = new edge_t*[batch_size];
    auto  h_new_ptr = new edge_t*[batch_size];
    auto    h_queue = new UpdateStr[batch_size];

    generateDeviceBatch(d_batch_src, d_batch_dest, batch_size,
                        BatchType::INSERT, prop);
    cuMalloc(d_queue_size);
    cuMalloc(d_degree, batch_size + 1);
    cuMalloc(d_queue, batch_size);      // allocate |queue| = num. unique src
    cuMalloc(d_old_ptrs, batch_size);
    cuMalloc(d_new_ptrs, batch_size);
    cuMemset0x00(d_degree);
    cuMemset0x00(d_queue_size);


    xlib::CubSortByKey<vid_t, vid_t> sort_cub(d_batch_src, d_batch_dest,
                                              batch_size, d_batch_src_sorted,
                                              d_batch_dest_sorted, _nV);

    xlib::CubRunLengthEncode<id_t> runlength_cub(d_batch_src_sorted, batch_size,
                                                 d_unique, d_counts);

    sort_cub.run();                        // batch sorting
    int num_items = runlength_cub.run();   // find src and src occurences

    CollectInfoForHost
        <<<xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE>>>
        (d_unique, d_counts, num_items, d_queue, d_queue_size);

    int h_queue_size;
    cuMemcpyToHost(d_queue_size, h_queue_size);

    if (h_queue_size > 0) {
        sendInfoToHost <<<xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE>>>
            (d_queue, h_queue_size, d_degree + 1, d_old_ptrs);

        CHECK_CUDA_ERROR
        cuMemcpyToHostAsync(d_old_ptrs, h_old_ptrs, h_queue_size);
        cuMemcpyToHostAsync(d_queue, h_queue, h_queue_size);

        for (int i = 0; i < h_queue_size; i++)
            mem_management.remove(h_old_ptrs[i], h_queue[i].old_degree);
        for (int i = 0; i < h_queue_size; i++)
            h_new_ptrs[i] = mem_management.insert(h_queue[i].new_degree).second;

        cuMemcpyToDeviceAsync(h_new_ptrs, d_new_ptrs, h_queue_size);

        //----------------------------------------------------------------------
        updateVertexBasicData
            <<<xlib::ceil_div<BLOCK_SIZE>(h_queue_size), BLOCK_SIZE>>>                        // update data structures
            (d_queue, h_queue_size, d_new_ptrs);
        //----------------------------------------------------------------------
        xlib::CubInclusiveSum<degree_t> prefixsum_cub(d_degree, h_queue_size);
        prefixsum_cub.run();

        degree_t* d_prefixsum = d_degree;     //alias
        int    prefixsum_size = h_queue_size; //alias

        degree_t prefixsum_total;           //get the total
        cuMemcpyToHost(d_prefixsum + prefixsum_size, prefixsum_total);
                                            //find partition size

        //----------------------------------------------------------------------
        const int     SMEM = xlib::SMemPerBlock<BLOCK_SIZE, degree_t>::value;
        int partition_size = xlib::ceil_div<SMEM>(prefixsum_total);

        bulkCopyAdjLists <<< partition_size, BLOCK_SIZE >>>
            (d_partition, d_prefixsum, d_old_ptrs, d_new_ptrs);

        bulkCopyAdjLists <<< partition_size, BLOCK_SIZE >>>
            (d_partition, d_prefixsum, d_old_ptrs, d_new_ptrs);
    }
*/
} // namespace custinger
