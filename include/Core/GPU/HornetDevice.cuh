/**
 * @brief High-level API to access to cuStinger data (Vertex, Edge)
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
#pragma once

#include "BasicTypes.hpp"                    //vid_t
#include "HostDevice.hpp"                    //HOST_DEVICE
#include "Core/DataLayout/DataLayoutDev.cuh" //BestLayoutDev
#include "Core/GPU/HornetTypes.cuh"          //Vertex

namespace hornet {
namespace gpu {

template<typename, typename> class HornetDevice;
/**
 * @internal
 * @brief The structure contanins all information for using the cuStinger data
 *        structure in the device
 */
template<typename... VertexTypes, typename... EdgeTypes>
class HornetDevice<TypeList<VertexTypes...>, TypeList<EdgeTypes...>> :
                           public BestLayoutDev<size_t, void*, VertexTypes...> {

    using VertexT = Vertex<TypeList<VertexTypes...>,
                           TypeList<EdgeTypes...>>;
public:
    using   EdgeT = Edge<TypeList<VertexTypes...>,
                         TypeList<EdgeTypes...>>;

    using edgeit_t = typename std::conditional<
                               xlib::IsVectorizable<vid_t, EdgeTypes...>::value,
                               AoSData<vid_t, EdgeTypes...>*, vid_t*>::type;

    explicit HornetDevice(vid_t nV, eoff_t nE, void* d_ptr, size_t pitch)
                          noexcept;

    HOST_DEVICE
    vid_t nV() const noexcept;

    HOST_DEVICE
    eoff_t nE() const noexcept;

    __device__ __forceinline__
    VertexT vertex(vid_t index);
private:
    ///@brief number of vertices in the graph
    vid_t  _nV { 0 };

    eoff_t _nE { 0 };
};

} // namespace gpu
} // namespace hornet

#include "impl/HornetDevice.i.cuh"
