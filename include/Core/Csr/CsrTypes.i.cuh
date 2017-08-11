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
#include "Core/cuStingerDevice.cuh" //cuStingerDevice

namespace custinger {

__device__ __forceinline__
Vertex::Vertex(cuStingerDevice& custinger, vid_t index) :
                                    _custinger(custinger),
                                    _id(index) {
    assert(index < custinger.nV());
    //xlib::SeqDev<VTypeSize> VTYPE_SIZE_D;

    auto offset = custinger.basic_data_ptr()[index];
    _id     = index;
    _degree = offset.y - offset.x;
    _offset = offset.x;
    /*#pragma unroll
    for (int i = 0; i < NUM_EXTRA_VTYPES; i++) {
        _vertex_ptrs[i] = custinger._d_vertex_ptrs[i] +
                          index * VTYPE_SIZE_D[i + 1];
    }*/
}

__device__ __forceinline__
Vertex::Vertex(cuStingerDevice& custinger) : _custinger(custinger) {}

__device__ __forceinline__
vid_t Vertex::id() const {
    return _id;
}

__device__ __forceinline__
degree_t Vertex::degree() const {
    return _degree;
}

__device__ __forceinline__
vid_t* Vertex::neighbor_ptr() const {
    return _custinger.edge_field_ptr<0>() + _offset;
}

__device__ __forceinline__
vid_t Vertex::neighbor_id(degree_t index) const {
    assert(index < _degree);
    return _custinger.edge_field_ptr<0>()[_offset + index];
}

template<typename T>
__device__ __forceinline__
WeightT* Vertex::edge_weight_ptr() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                  "weight is not part of edge type list");
    return _custinger.edge_field_ptr<1>() + _offset;
}

template<int INDEX>
__device__ __forceinline__
typename std::tuple_element<INDEX, VertexTypes>::type
Vertex::field() const {
    return _custinger.vertex_field_ptr<INDEX>()[_id];
}

__device__ __forceinline__
Edge Vertex::edge(degree_t index) const {
    return Edge(_custinger, _offset + index);
}

//==============================================================================
//==============================================================================

__device__ __forceinline__
Edge::Edge(cuStingerDevice& custinger, eoff_t offset) :
                _custinger(custinger),
                _offset(offset),
                _dst_id(_custinger.edge_field_ptr<0>()[offset]),
                _tmp_vertex(custinger),
                _src_vertex(_tmp_vertex) {}

__device__ __forceinline__
Edge::Edge(cuStingerDevice& custinger, eoff_t offset, vid_t src_id) :
                _custinger(custinger),
                _offset(offset),
                _dst_id(_custinger.edge_field_ptr<0>()[offset]),
                _src_id(src_id),
                _tmp_vertex(custinger, src_id),
                _src_vertex(_tmp_vertex) {}

__device__ __forceinline__
Edge::Edge(cuStingerDevice& custinger, eoff_t offset, Vertex& src_vertex) :
                _custinger(custinger),
                _offset(offset),
                _dst_id(_custinger.edge_field_ptr<0>()[offset]),
                _src_id(src_vertex.id()),
                _tmp_vertex(custinger),
                _src_vertex(src_vertex) {}

__device__ __forceinline__
vid_t Edge::src_id() const {
    return _src_id;
}

__device__ __forceinline__
vid_t Edge::dst_id() const {
    return _dst_id;
}

__device__ __forceinline__
Vertex& Edge::src() const {
    return _src_vertex;
}

__device__ __forceinline__
Vertex Edge::dst() const {
    return Vertex(_custinger, _dst_id);
}

template<typename T>
__device__ __forceinline__
WeightT Edge::weight() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                  "weight is not part of edge type list");
    return _custinger.edge_field_ptr<1>()[_offset];
}

template<typename T>
__device__ __forceinline__
void Edge::set_weight(WeightT weight) {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                  "weight is not part of edge type list");
    _custinger.edge_field_ptr<1>()[_offset] = weight;
}

template<typename T>
__device__ __forceinline__
TimeStamp1T Edge::time_stamp1() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 2,
                  "time_stamp1 is not part of edge type list");
    return _custinger.edge_field_ptr<2>()[_offset];
}

template<typename T>
__device__ __forceinline__
TimeStamp2T Edge::time_stamp2() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 2,
                  "time_stamp2 is not part of edge type list");
    return _custinger.edge_field_ptr<3>()[_offset];
}

template<int INDEX>
__device__ __forceinline__
typename std::tuple_element<INDEX, EdgeTypes>::type
Edge::field() const {
    return _custinger.edge_field_ptr<INDEX>()[_offset];
}

} // namespace custinger
