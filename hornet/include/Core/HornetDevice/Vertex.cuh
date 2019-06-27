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
#ifndef VERTEX_CUH
#define VERTEX_CUH

#include "../Conf/MemoryManagerConf.cuh"
#include "../Conf/HornetConf.cuh"
#include <type_traits>

namespace hornet {

template <typename, typename,
         typename = VID_T, typename = DEGREE_T>
         class Vertex;

template<typename, typename, typename, typename> class HornetDevice;
template<typename, typename, typename, typename> class Edge;

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
class Vertex<
    TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>,
    vid_t, degree_t> {

    template <typename, typename, typename, typename> friend class Edge;
    template <typename, typename, typename, typename> friend class HornetDevice;

    public:

    using HornetDeviceT = HornetDevice<
        TypeList<VertexMetaTypes...>,
        TypeList<EdgeMetaTypes...>,
        vid_t, degree_t>;

    using EdgeT = Edge<
        TypeList<VertexMetaTypes...>,
        TypeList<EdgeMetaTypes...>,
        vid_t, degree_t>;

    private:

    HornetDeviceT&  _hornet;

    vid_t           _id;

    HOST_DEVICE
    Vertex(HornetDeviceT& hornet, const vid_t id);

    public:
    HOST_DEVICE
    vid_t id(void) const;

    HOST_DEVICE
    degree_t degree(void) const;

    HOST_DEVICE
    degree_t limit(void) const;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<
        (N < sizeof...(VertexMetaTypes)),
        typename xlib::SelectType<N, VertexMetaTypes&...>::type>::type
    field(void) const;

    HOST_DEVICE
    EdgeT
    edge(const degree_t index) const;

    HOST_DEVICE
    degree_t edges_per_block(void) const;

    HOST_DEVICE
    degree_t vertex_offset(void) const;

    HOST_DEVICE
    xlib::byte_t* edge_block_ptr(void) const;

    HOST_DEVICE
    void set_degree(degree_t new_degree) const;

    HOST_DEVICE
    vid_t* neighbor_ptr(void) const;

};

}
#include "impl/Vertex.i.cuh"
#endif
