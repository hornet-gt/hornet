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
 *
 * @file
 */
#pragma once

#include "Core/cuStingerConfig.hpp"
#include "Core/MemoryManagement.hpp"
#include <cstddef>                      //size_t

using xlib::byte_t;

/**
 * @brief
 */
namespace cu_stinger {

/**
 * @brief Main cuStinger class
 */
class cuStinger {
public:
    /**
     * @brief default costructor
     * @param[in] num_vertices
     * @param[in] num_edges
     * @param[in] csr_offset
     * @param[in] csr_edges
     */
    cuStinger(size_t num_vertices, size_t num_edges,
              const off_t* csr_offset, const id_t* csr_edges) noexcept;

    /**
     * @brief decostructor
     */
    ~cuStinger() noexcept;

    /**
     * @brief Insert additional vertex data
     * @param[in] vertex_data head of the list of vertex data
     * @param[in] args tail of the list of vertex data
     * @remark the types of the input arrays must be equal to the type List
     *         for vertices specified in the *cuStingerConf.hpp* file
     */
    template<unsigned INDEX = 0, typename T, typename... TArgs>
    void insertVertexData(const T* vertex_data, TArgs... args) noexcept;

    /**
     * @brief Insert additional edge data
     * @param[in] edge_data head of the list of edge data
     * @param[in] args tail of the list of edge data
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *cuStingerConf.hpp* file
     */
    template<unsigned INDEX = 0, typename T, typename... TArgs>
    void insertEdgeData(const T* edge_data, TArgs... args) noexcept;

    /**
     * @brief Initialize the data structure
     * @details
     */
    void initialize() noexcept;

    /**
     * @brief Initialize the data structure
     * @details
     */
    void print() noexcept;

private:
    MemoryManagement mem_management;

    byte_t* _vertex_data_ptr[ NUM_EXTRA_VTYPES ];
    byte_t*   _edge_data_ptr[ NUM_ETYPES ];

    size_t       _nV;
    size_t       _nE;
    const off_t* _csr_offset;
    vertex_t*    _d_nodes        { nullptr };
    bool         _vertex_init    { false };
    bool         _edge_init      { false };
    bool         _custinger_init { false };


    template<unsigned INDEX>
    void insertVertexData() noexcept;

    template<unsigned INDEX>
    void insertEdgeData() noexcept;

    void initializeGlobal() noexcept;
};

} // namespace cu_stinger

#include "cuStinger.i.hpp"
