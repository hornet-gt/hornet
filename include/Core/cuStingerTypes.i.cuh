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
namespace cu_stinger {

__device__ __forceinline__
Vertex::Vertex(id_t index) noexcept {
    xlib::SeqDev<VTypeSize> VTYPE_SIZE_D;
    _ptrs[0] = d_ptrs[0] + index * VTYPE_SIZE_D[0];   //degree
    _ptrs[1] = d_ptrs[1] + index * VTYPE_SIZE_D[1];   //limit
    auto ptr = d_ptrs[2] + index * VTYPE_SIZE_D[2];   //edge
    _ptrs[2] = reinterpret_cast<byte_t*>(*reinterpret_cast<edge_t**>(ptr));
    //#pragma unroll
    //for (int i = 0; i < NUM_EXTRA_VTYPES; i++)      //Fused multiply–add
    //    _ptrs[i + 3] = d_ptrs[i + 3] + index * VTYPE_SIZE2[i];//EXTRA_VTYPE_SIZE[i];
}

__device__ __forceinline__
degree_t Vertex::degree() const noexcept {
    return *reinterpret_cast<degree_t*>(_ptrs[0]);
}

__device__ __forceinline__
Edge Vertex::edge(degree_t index) const noexcept {
    return Edge(_ptrs[2], index);
}

//==============================================================================

__device__ __forceinline__
Edge::Edge(byte_t* block_ptr, off_t index) noexcept {
     _ptrs[0] = block_ptr + index * sizeof(id_t);
    //#pragma unroll
    //for (int i = 0; i < NUM_ETYPES; i++)
    //    _ptrs[i] = block_ptr + index * ETYPE_SIZE[i];       //Fused multiply–add
}

__device__ __forceinline__
id_t Edge::dst() const noexcept {
    return *reinterpret_cast<id_t*>(_ptrs[0]);
}

template<typename T>
__device__ __forceinline__
typename std::tuple_element<(NUM_ETYPES > 1 ? 1 : 0), edge_t>::type
Edge::weight() const noexcept {
    static_assert(!std::is_same<T, void>::value,
                  "weight is not part of edge type list");
    const int N = NUM_ETYPES > 1 ? 1 : 0;
    using     R = typename std::tuple_element<N, edge_t>::type;
    return *reinterpret_cast<R*>(_ptrs[N]);
}

//==============================================================================


} // namespace cu_stinger
