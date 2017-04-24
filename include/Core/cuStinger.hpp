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

#include "Core/RawTypes.hpp"
#include "Core/MemoryManagement.hpp"
#include <cstddef>                      //size_t

/**
 * @brief
 */
namespace cu_stinger {

class cuStinger;

/**
 * @brief cuStinger initialization class
 */
class cuStingerInit {
    friend cuStinger;
public:
    /**
     * @brief default costructor
     * @param[in] num_vertices number of vertices
     * @param[in] num_edges number of edges
     * @param[in] csr_offsets csr offsets array
     * @param[in] csr_edges csr edges array
     */
    explicit cuStingerInit(size_t num_vertices, size_t num_edges,
                           const off_t* csr_offsets, const id_t* csr_edges)
                           noexcept;

    /**
     * @brief Insert additional vertex data
     * @param[in] vertex_data list of vertex data array
     * @remark the types of the input arrays must be equal to the type List
     *         for vertices specified in the *config.inc* file
     * @details **Example**
     *         @code{.cpp}
     *             int* array1 = ...;
     *             float* array2 = ...;
     *             cuStingerInit custinger_init(...);
     *             custinger_init.insertVertexData(array1, array2);
     *         @endcode
     */
    template<typename... TArgs>
    void insertVertexData(TArgs... vertex_data) noexcept;

    /**
     * @brief Insert additional edge data
     * @param[in] edge_data list of vertex data array
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *config.inc* file
     * @see ::insertVertexData
     */
    template<typename... TArgs>
    void insertEdgeData(TArgs... edge_data) noexcept;

    /**
     *
     */
    size_t nV() const noexcept;

    /**
     *
     */
    size_t nE() const noexcept;

    /**
     *
     */
    const off_t* csr_offsets() const noexcept;

private:
    /**
     * @internal
     * @brief Array of pointers of the *additional* vertex data
     */
    byte_t* _vertex_data_ptrs[ NUM_EXTRA_VTYPES ] = {};

    /**
     * @internal
     * @brief Array of pointers of the *all* edge data
     */
    byte_t*   _edge_data_ptrs[ NUM_ETYPES ] = {};

    size_t       _nV;
    size_t       _nE;
    const off_t* _csr_offsets;
};

//==============================================================================

/**
 * @brief Main cuStinger class
 */
class cuStinger {
public:
    /**
     * @brief default costructor
     * @param[in] custinger_init cuStinger initilialization data structure
     */
    explicit cuStinger(const cuStingerInit& custinger_init) noexcept;

    /**
     * @brief decostructor
     */
    ~cuStinger() noexcept;

    /**
     * @brief print the graph directly from the device
     * @warning this function should be applied only on small graphs
     */
    void print() noexcept;

    /**
     * @brief Check the consistency of the device data structure with the host
     *        data structure provided in the input
     * @details revert the initilization process to rebuild the device data
     *          structure on the host
     */
    void check_consistency(const cuStingerInit& custinger_init) const noexcept;

private:
    MemoryManagement mem_management;

    /**
     * @internal
     * @brief device pointer for *all* vertex data
     *        (degree and edge pointer included)
     */
    byte_t*      _d_vertices { nullptr };
    size_t       _nV;
    size_t       _nE;

    /**
     * @internal
     * @brief copy the vertex data pointers to the __constant__ memory
     */
    void initializeVertexGlobal(byte_t* (&vertex_data_ptrs)[NUM_VTYPES])
                                noexcept;
};

//==============================================================================

/**
 * @brief Batch Property
 */
class BatchProperty {
public:
    /**
     * @brief default costructor
     * @param[in] sort the edge batch is sorted in lexicographic order
     *            (source, destination)
     * @param[in] weighted_distr generate a batch by using a random weighted
     *            distribution based on the degree of the vertices
     * @param[in] print print the batch on the standard output
     */
    explicit BatchProperty(bool           sort = false,
                           bool weighted_distr = false,
                           bool          print = false) noexcept;
private:
    bool _sort, _print, _weighted_distr;
};

/**
 * @brief Batch update class
 */
class BatchUpdate {
    friend cuStinger;
public:
    /**
     * @brief default costructor
     * @param[in] batch_size number of edges of the batch
     */
    explicit BatchUpdate(size_t batch_size) noexcept;

    /**
     * @brief Insert additional edge data
     * @param[in] edge_data list of edge data. The list must contains atleast
     *            the source and the destination arrays (id_t type)
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *config.inc* file
     * @see ::insertVertexData
     */
    template<typename... TArgs>
    void insertEdgeData(TArgs... edge_data) noexcept;

private:
    byte_t*       _edge_data_ptrs[ NUM_ETYPES + 1 ]; //+1 for source ids
    size_t        _batch_size;
};

} // namespace cu_stinger

#include "cuStinger.i.hpp"
