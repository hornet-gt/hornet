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

inline BatchInit::BatchInit(const vid_t* src_array, const vid_t* dst_array,
                            int batch_size) noexcept :
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

    for (int i = 0; i < NUM_ETYPES + 1; i++) {
        if (batch_init.edge_ptrs(i) == nullptr)
            ERROR("Edge data not initializated");
    }
    byte_t* ptr;
    cuMalloc(ptr, _batch_pitch * (sizeof(vid_t) + sizeof(edge_t)));
    _d_edge_ptrs[0] = ptr;
    cuMemcpyToDeviceAsync(batch_init.edge_ptrs(0), _batch_size * sizeof(vid_t),
                          _d_edge_ptrs[0]);
    ptr += _batch_pitch * sizeof(vid_t);
    for (int i = 0; i < NUM_ETYPES; i++) {
        _d_edge_ptrs[i + 1] = ptr;
        cuMemcpyToDeviceAsync(batch_init.edge_ptrs(i + 1),
                              _batch_size * ETYPE_SIZE[i], _d_edge_ptrs[i + 1]);
        ptr += _batch_pitch * ETYPE_SIZE[i];
    }
    for (int i = 0; i < NUM_ETYPES + 1; i++)
        std::cout << (int*) _d_edge_ptrs[i] << std::endl;
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

__host__ __device__ __forceinline__
vid_t* BatchUpdate::src_ptr() const noexcept {
    return reinterpret_cast<vid_t*>(_d_edge_ptrs[0]);
}

__host__ __device__ __forceinline__
vid_t* BatchUpdate::dst_ptr() const noexcept {
    return reinterpret_cast<vid_t*>(_d_edge_ptrs[1]);
}

__device__ __forceinline__
Edge BatchUpdate::edge(int index) const noexcept {
    return Edge(_d_edge_ptrs[1], index, _batch_pitch);
}

template<int INDEX>
__device__ __forceinline__
typename std::tuple_element<INDEX, VertexTypes>::type
BatchUpdate::field(int index) const noexcept {
    //_batch_pitch
}

#endif

} // namespace custinger
