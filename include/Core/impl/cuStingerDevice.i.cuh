/**
 * @brief High-level API to access to cuStinger data (Vertex, Edge)
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
 *
 * @file
 */
#if defined(__NVCC__)
    #include "Core/cuStingerTypes.cuh"
#endif

namespace custinger {

inline
cuStingerDevice::cuStingerDevice(vid_t nV, eoff_t nE,
                                 byte_t* const (&d_vertex_ptrs)[NUM_VTYPES])
                                 noexcept {
    _nV = nV;
    _nE = nE;
    std::copy(d_vertex_ptrs, d_vertex_ptrs + NUM_VTYPES, _d_vertex_ptrs);
}

#if defined(__NVCC__)

__device__ __forceinline__
vid_t cuStingerDevice::nV() const noexcept {
    return _nV;
}

__device__ __forceinline__
eoff_t cuStingerDevice::nE() const noexcept {
    return _nE;
}

__device__ __forceinline__
Vertex cuStingerDevice::vertex(vid_t index) noexcept {
    return Vertex(*this, index);
}

__device__ __forceinline__
VertexBasicData* cuStingerDevice::basic_data_ptr() const noexcept {
    return reinterpret_cast<VertexBasicData*>(_d_vertex_ptrs[0]);
}

template<int INDEX>
__device__ __forceinline__
typename std::tuple_element<INDEX, vertex_t>::type*
cuStingerDevice::vertex_field_ptr() const {
    using T = typename std::tuple_element<INDEX, vertex_t>::type;
    return *reinterpret_cast<T*>(_d_vertex_ptrs[INDEX]);
}

///only CSR
template<int INDEX>
__device__ __forceinline__
IndexT<INDEX>* cuStingerDevice::edge_field_ptr() const {
    using T = typename std::tuple_element<INDEX, edge_t>::type;
    return reinterpret_cast<T*>(_d_edge_ptrs[INDEX]);
}
#endif

} // namespace custinger
