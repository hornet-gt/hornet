/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Core/cuStingerConf.hpp"
#include "Core/MemoryManagement.hpp"
#include <cstddef>                      //size_t

/**
 *
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
     * @param[in] list of arrays containing the vertex data
     * @remark the types of the input arrays must be equal to the type List
     *         for vertices specified in the *cuStingerConf.hpp* file
     */
    template<unsigned INDEX = 1, typename T, typename... TArgs>
    void insertVertexData(T* vertex_data, TArgs... args) noexcept;

    /**
     * @brief Insert additional edge data
     * @param[in] list of arrays containing the edge data
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *cuStingerConf.hpp* file
     */
    template<unsigned INDEX = 1, typename T, typename... TArgs>
    void insertEdgeData(T* edge_data, TArgs... args) noexcept;

    /**
     * @brief Initialize the data structure
     * @detail
     */
    void initialize() noexcept;

private:
    static const unsigned NUM_VERTEX_TYPES = std::tuple_size<VertexTypes>::value;
    static const unsigned   NUM_EDGE_TYPES = std::tuple_size<EdgeTypes>::value;

    MemoryManagement mem_management;

    void* _vertex_data_ptr[ NUM_VERTEX_TYPES ];
    void*   _edge_data_ptr[ NUM_EDGE_TYPES ];

    size_t      _nV;
    size_t      _nE;
    const id_t* _csr_edges { nullptr };
    degree_t*   _degrees   { nullptr };
    degree_t*   _limits    { nullptr };

    template<unsigned INDEX>
    void insertVertexData() noexcept;

    template<unsigned INDEX>
    void insertEdgeData() noexcept;
};

} // namespace cu_stinger

#include "cuStinger.i.hpp"
