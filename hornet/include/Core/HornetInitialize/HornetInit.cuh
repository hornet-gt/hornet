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
 *
 * @file
 */
#ifndef HORNET_INIT_CUH
#define HORNET_INIT_CUH

#include "../Conf/HornetConf.cuh"

/**
 * @brief The namespace contanins all classes and methods related to the
 *        Hornet data structure
 */
namespace hornet {

template <
         typename = VID_T,
         typename = EMPTY, typename = EMPTY, typename = DEGREE_T>
         class HornetInit;

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
class HornetInit<
    vid_t,
    TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, degree_t> {

    vid_t                       _nV          { 0 };
    degree_t                    _nE          { 0 };
    //const degree_t *            _csr_offsets { nullptr };
    //const vid_t *               _csr_edges   { nullptr };

    SoAPtr<degree_t const, VertexMetaTypes const...> _vertex_data;
    SoAPtr<vid_t const, EdgeMetaTypes const...>        _edge_data;

public:

    HornetInit(
            const vid_t num_vertices, const degree_t num_edges,
            const degree_t* csr_offsets,
            const vid_t*    csr_edges) noexcept;

    void insertEdgeData(EdgeMetaTypes const *... edge_meta_data) noexcept;

    template <unsigned N>
    void insertEdgeData(typename xlib::SelectType<N, EdgeMetaTypes const *...>::type edge_meta_data) noexcept;

    void insertVertexData(VertexMetaTypes const *... vertex_meta_data) noexcept;

    template <unsigned N>
    void insertVertexData(typename xlib::SelectType<N, VertexMetaTypes const *...>::type vertex_meta_data) noexcept;

    vid_t nV() const noexcept;

    degree_t nE() const noexcept;

    const degree_t* csr_offsets() const noexcept;

    const vid_t* csr_edges() const noexcept;

    SoAPtr<degree_t const, VertexMetaTypes const...> vertex_data_ptr(void) const noexcept;

    SoAPtr<vid_t const, EdgeMetaTypes const...>   edge_data_ptr(void) const noexcept;

};

} // namespace hornet

#include "HornetInit.i.cuh"
#endif
