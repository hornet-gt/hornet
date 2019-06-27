/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
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
#include "Host/Metaprogramming.hpp"

namespace hornet {

#define HORNET_INIT HornetInit<vid_t,\
                               TypeList<VertexMetaTypes...>,\
                               TypeList<EdgeMetaTypes...>, degree_t>

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline
HORNET_INIT::
HornetInit(
        const vid_t num_vertices, const degree_t num_edges,
        const degree_t * const csr_offsets, const vid_t * const csr_edges) noexcept :
            _nV(num_vertices),
            _nE(num_edges) {
    _vertex_data.template set<0>(csr_offsets);
    _edge_data.template set<0>(csr_edges);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline void
HORNET_INIT::
insertEdgeData(EdgeMetaTypes const *... edge_meta_data) noexcept {
    SoAPtr<vid_t const, EdgeMetaTypes const...> e_m(_edge_data.template get<0>(), edge_meta_data...);
    _edge_data = e_m;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <unsigned N>
inline void
HORNET_INIT::
insertEdgeData(typename xlib::SelectType<N, EdgeMetaTypes const *...>::type edge_meta_data) noexcept {
    _edge_data.template set<N+1>(edge_meta_data);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline void
HORNET_INIT::
insertVertexData(VertexMetaTypes const *... vertex_meta_data) noexcept {
    SoAPtr<degree_t const, VertexMetaTypes const...> v_m(_vertex_data.template get<0>(), vertex_meta_data...);
    _vertex_data = v_m;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <unsigned N>
inline void
HORNET_INIT::
insertVertexData(typename xlib::SelectType<N, VertexMetaTypes const *...>::type vertex_meta_data) noexcept {
    _vertex_data.template set<N+1>(vertex_meta_data);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline vid_t
HORNET_INIT::
nV() const noexcept {
    return _nV;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline degree_t
HORNET_INIT::
nE() const noexcept {
    return _nE;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline const degree_t*
HORNET_INIT::
csr_offsets() const noexcept {
    return _vertex_data.template get<0>();
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline const vid_t*
HORNET_INIT::
csr_edges() const noexcept {
    return _edge_data.template get<0>();
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline SoAPtr<degree_t const, VertexMetaTypes const...>
HORNET_INIT::
vertex_data_ptr(void) const noexcept {
    return _vertex_data;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
inline SoAPtr<vid_t const, EdgeMetaTypes const...>
HORNET_INIT::
edge_data_ptr(void) const noexcept {
    return _edge_data;
}

} // namespace hornet
