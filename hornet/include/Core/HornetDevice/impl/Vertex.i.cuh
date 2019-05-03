/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
namespace hornet {

#define VERTEX Vertex<TypeList<VertexMetaTypes...>,\
                                   TypeList<EdgeMetaTypes...>,\
                                   vid_t,\
                                   degree_t>

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
VERTEX::
Vertex(HornetDeviceT& hornet, const vid_t id) :
    _hornet(hornet), _id(id) {}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
vid_t
VERTEX::
id(void) const {
    return _id;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
degree_t
VERTEX::
degree(void) const {
    return (_hornet.get_vertex_data().template get<0>())[_id];
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
xlib::byte_t*
VERTEX::
edge_block_ptr(void) const {
    return (_hornet.get_vertex_data().template get<1>())[_id];
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
degree_t
VERTEX::
vertex_offset(void) const {
    return (_hornet.get_vertex_data().template get<2>())[_id];
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
degree_t
VERTEX::
edges_per_block(void) const {
    return (_hornet.get_vertex_data().template get<3>())[_id];
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
degree_t
VERTEX::
limit(void) const {
    return xlib::max(static_cast<degree_t>(MIN_EDGES_PER_BLOCK),
            PREFER_FASTER_UPDATE ? xlib::roundup_pow2(degree() + 1) :
            xlib::roundup_pow2(degree()));
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<
    (N < sizeof...(VertexMetaTypes)),
    typename xlib::SelectType<N, VertexMetaTypes&...>::type>::type
VERTEX::
field(void) const {
    return (_hornet.get_vertex_data().template get<N+4>())[_id];
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
VERTEX::EdgeT
VERTEX::
edge(const degree_t index) const {
    return VERTEX::EdgeT(_hornet, _id, index,
            edge_block_ptr(),
            vertex_offset(),
            edges_per_block());
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
void
VERTEX::
set_degree(degree_t new_degree) const {
    (_hornet.get_vertex_data().template get<0>())[_id] = new_degree;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
vid_t*
VERTEX::
neighbor_ptr(void) const {
    return reinterpret_cast<vid_t*>(edge_block_ptr()) + vertex_offset();
}

#undef VERTEX
}
