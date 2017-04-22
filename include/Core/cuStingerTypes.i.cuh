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
#include "Core/cuStingerGlobalSpace.cuh"

namespace cu_stinger {

__device__ __forceinline__
Vertex::Vertex(id_t index) : _id(index) {
    assert(index < d_nV);
    xlib::SeqDev<VTypeSize> VTYPE_SIZE_D;
    _vertex_ptr     = d_vertex_basic_ptr + index;
    auto basic_data = *_vertex_ptr;

    _degree   = basic_data.degree;
    _limit    = detail::limit(_degree);
    _edge_ptr = basic_data.edge_ptr;
    #pragma unroll
    for (int i = 0; i < NUM_EXTRA_VTYPES; i++)
        _ptrs[i] = d_vertex_data_ptrs[i] + index * VTYPE_SIZE_D[i + 1];
}
/*
__device__ __forceinline__
Vertex::Vertex() {}*/

__device__ __forceinline__
degree_t Vertex::id() const {
    return _id;
}

__device__ __forceinline__
degree_t Vertex::degree() const {
    return _degree;
}

__device__ __forceinline__
Edge Vertex::edge(degree_t index) const {
    return Edge(_edge_ptr, index, _limit);
}

__device__ __forceinline__
degree_t Vertex::limit() const {
    return _limit;
}

__device__ __forceinline__
degree_t* Vertex::degree_ptr() {
    return reinterpret_cast<degree_t*>(_vertex_ptr + sizeof(byte_t*));
}

template<int INDEX>
__device__ __forceinline__
typename std::tuple_element<INDEX, VertexTypes>::type
Vertex::field() const {
    using T = typename std::tuple_element<INDEX, VertexTypes>::type;
    return *reinterpret_cast<T*>(_ptrs[INDEX]);
}

//------------------------------------------------------------------------------

namespace detail {

template<int INDEX = 0>
__device__ __forceinline__
void store_edge(byte_t* const (&load_ptrs)[NUM_EXTRA_ETYPES],
                byte_t*      (&store_ptrs)[NUM_EXTRA_ETYPES]) {
    using T = typename std::tuple_element<INDEX, EdgeTypes>::type;
    *reinterpret_cast<T*>(store_ptrs) = *reinterpret_cast<const T*>(load_ptrs);
    store_edge<INDEX + 1>(load_ptrs, store_ptrs);
}
template<>
__device__ __forceinline__
void store_edge<NUM_EXTRA_ETYPES>(byte_t* const (&)[NUM_EXTRA_ETYPES],
                                  byte_t*       (&)[NUM_EXTRA_ETYPES]) {}
} // namespace detail

//------------------------------------------------------------------------------

__device__ __forceinline__
void Vertex::store(const Edge& edge, degree_t index) {
    Edge to_replace(_edge_ptr, index, _limit);

    reinterpret_cast<id_t*>(_edge_ptr)[index] = edge.dst();
    detail::store_edge(edge._ptrs, to_replace._ptrs);
}

//==============================================================================
//==============================================================================

__device__ __forceinline__
Edge::Edge(byte_t* edge_ptr, degree_t index, degree_t limit) {
    //Edge Type Sizes Prefixsum
    xlib::SeqDev<ETypeSizePS> ETYPE_SIZE_PS_D;

    _dst = reinterpret_cast<id_t*>(edge_ptr)[index];
    #pragma unroll
    for (int i = 0; i < NUM_EXTRA_ETYPES; i++)
        _ptrs[i] = edge_ptr + limit * ETYPE_SIZE_PS_D[i + 1];
}

__device__ __forceinline__
id_t Edge::dst() const {
    return _dst;
}

template<typename T>
__device__ __forceinline__
typename Edge::WeightT Edge::weight() const {
    static_assert(!std::is_same<T, void>::value,
                  "weight is not part of edge type list");
    return *reinterpret_cast<WeightT*>(_ptrs[0]);
}

template<typename T>
__device__ __forceinline__
typename Edge::TimeStamp1T Edge::time_stamp1() const {
    static_assert(!std::is_same<T, void>::value,
                  "weight is not part of edge type list");
    return *reinterpret_cast<TimeStamp1T*>(_ptrs[1]);
}

template<typename T>
__device__ __forceinline__
typename Edge::TimeStamp2T Edge::time_stamp2() const {
    static_assert(!std::is_same<T, void>::value,
                  "weight is not part of edge type list");
    return *reinterpret_cast<TimeStamp2T*>(_ptrs[2]);
}

template<int INDEX>
__device__ __forceinline__
typename std::tuple_element<INDEX, EdgeTypes>::type
Edge::field() const  {
    using T = typename std::tuple_element<INDEX, EdgeTypes>::type;
    return *reinterpret_cast<T*>(_ptrs[INDEX]);
}

//==============================================================================


} // namespace cu_stinger
