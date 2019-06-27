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

#define EDGE Edge<TypeList<VertexMetaTypes...>,\
                                   TypeList<EdgeMetaTypes...>,\
                                   vid_t,\
                                   degree_t>

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
EDGE::Edge(
        HornetDeviceT& hornet, const vid_t src_id, const degree_t index,
        xlib::byte_t* const edge_block_ptr,
        const degree_t vertex_offset,
        const degree_t edges_per_block) :
    _hornet(hornet),
    _src_id(src_id),
    _index(index + vertex_offset),
    _ptr(edge_block_ptr, edges_per_block) {}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
vid_t
EDGE::src_id(void) const {
    return _src_id;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
vid_t
EDGE::dst_id(void) const {
    return _ptr.template get<0>()[_index];
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
EDGE::VertexT
EDGE::src(void) const {
    return _hornet.vertex(_src_id);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
EDGE::VertexT
EDGE::dst(void) const {
    return _hornet.vertex(dst_id());
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<
    (N < sizeof...(EdgeMetaTypes)),
    typename xlib::SelectType<N, EdgeMetaTypes&...>::type>::type
EDGE::field(void) const {
    return _ptr.template get<N+1>()[_index];
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HOST_DEVICE
EDGE&
EDGE::operator=(const EDGE& source_edge) noexcept {
    RecursiveAssign<0, sizeof...(EdgeMetaTypes)>::assign(source_edge._ptr, source_edge._index, _ptr, _index);
    return *this;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename SRef>
HOST_DEVICE
EDGE&
EDGE::operator=(const SRef& source_edge) noexcept {
    RecursiveAssign<0, sizeof...(EdgeMetaTypes)>::assign(source_edge, _ptr, _index);
    return *this;
}

#undef EDGE
}
