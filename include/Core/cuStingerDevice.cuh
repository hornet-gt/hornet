/**
 * @brief High-level API to access to cuStinger data (Vertex, Edge)
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
 *
 * @file
 */
#pragma once

#include "Core/RawTypes.hpp"

namespace custinger {

class Vertex;

/**
 * @internal
 * @brief The structure contanins all information for using the cuStinger data
 *        structure in the device
 */
class cuStingerDevice {
    friend class Vertex;
public:
    explicit cuStingerDevice(vid_t nV, eoff_t nE,
                             byte_t* const (&d_vertex_ptrs)[NUM_VTYPES])
                             noexcept;

    explicit cuStingerDevice(vid_t nV, eoff_t nE,
                             byte_t* const (&d_vertex_ptrs)[NUM_VTYPES],
                             byte_t* const (&d_edge_ptrs)[NUM_VTYPES])
                             noexcept;

    HOST_DEVICE
    vid_t nV() const noexcept;

    HOST_DEVICE
    eoff_t nE() const noexcept;

#if defined(__NVCC__)
    __device__ __forceinline__
    Vertex vertex(vid_t index) noexcept;

    __device__ __forceinline__
    VertexBasicData* basic_data_ptr() const noexcept;

    template<int INDEX>
    __device__ __forceinline__
    typename std::tuple_element<INDEX, vertex_t>::type*
    vertex_field_ptr() const;

    ///only CSR
    template<int INDEX>
    __device__ __forceinline__
    IndexT<INDEX>* edge_field_ptr() const;
#endif
private:
    /**
     * @brief array of pointers to vertex data
     * @detail the first element points to the structure that contanins the
     *         edge pointer and the degree of the vertex
     */
    byte_t* _d_vertex_ptrs[NUM_VTYPES] = {};
    byte_t* _d_edge_ptrs[NUM_VTYPES]   = {};    //CSR

    ///@brief number of vertices in the graph
    vid_t   _nV { 0 };

    eoff_t  _nE { 0 };
};

} // namespace custinger

#include "impl/cuStingerDevice.i.cuh"
