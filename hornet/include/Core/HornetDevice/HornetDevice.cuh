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
#ifndef HORNET_DEVICE_CUH
#define HORNET_DEVICE_CUH

#include "../Conf/HornetConf.cuh"
#include "../SoA/SoAPtr.cuh"
#include "../SoA/SoAData.cuh"//TODO : Remove
#include "Vertex.cuh"
#include "Edge.cuh"
#include <type_traits>

namespace hornet {

template <typename, typename,
         typename = VID_T, typename = DEGREE_T>
         class HornetDevice;

template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
class HornetDevice<
    TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t> {

    template <typename, typename, typename, typename> friend class Vertex;
    template <typename, typename, typename, typename> friend class Edge;

    vid_t       _nV { 0 };

    degree_t    _nE { 0 };

    SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...> _vertex_data;

    public:

    using VertexT = Vertex<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>;

    using VertexType = vid_t;

    using DegreeType = degree_t;

    HOST_DEVICE
    SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>
    get_vertex_data(void) noexcept;

    explicit HornetDevice(
        vid_t nV,
        degree_t nE,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>& vertex_data) noexcept;

    HOST_DEVICE
    vid_t nV(void) noexcept;

    HOST_DEVICE
    degree_t nE(void) noexcept;

    HOST_DEVICE
    VertexT vertex(const vid_t index) noexcept;
};


}

#include "impl/HornetDevice.i.cuh"
#endif
