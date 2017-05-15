/**
 * @brief cuStinger, cuStingerInit, BatchUpdatem and BatchProperty classes
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
#include "Core/MemoryManager.hpp"
#include <cstddef>                      //size_t

/**
 * @brief The namespace contanins all classes and methods related to the
 *        cuStinger data structure
 */
namespace custinger {

class cuStinger;


/**
 * @brief cuStinger initialization class
 */
class cuStingerInit {
    friend cuStinger;
public:
    /**
     * @brief Default costructor
     * @param[in] num_vertices number of vertices
     * @param[in] num_edges number of edges
     * @param[in] csr_offsets csr offsets array
     * @param[in] csr_edges csr edges array
     */
    explicit cuStingerInit(size_t num_vertices, size_t num_edges,
                           const eoff_t* csr_offsets, const vid_t* csr_edges)
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
     * @brief number of vertices in the graph passed to the costructor
     * @return number of vertices in the graph
     */
    size_t nV() const noexcept;

    /**
     * @brief number of edges in the graph passed to the costructor
     * @return number of edges in the graph
     */
    size_t nE() const noexcept;

    /**
     * @brief CSR offsets of the graph passed to the costructor
     * @return constant pointer to the csr offsets
     */
    const eoff_t* csr_offsets() const noexcept;

    /**
     * @brief CSR edges of the graph passed to the costructor
     * @return constant pointer to the csr edges
     */
    const vid_t* csr_edges() const noexcept;

private:
    /**
     * @internal
     * @brief Array of pointers of the *all* vertex data
     */
    const byte_t* _vertex_data_ptrs[ NUM_VTYPES ] = {};

    /**
     * @internal
     * @brief Array of pointers of the *all* edge data
     */
    const byte_t* _edge_data_ptrs[ NUM_ETYPES ] = {};
    size_t        _nV;
    size_t        _nE;
};

//==============================================================================

/**
 * @internal
 * @brief The structure contanins all information to use the cuStinger data
 *        structure in the device
 */
struct cuStingerDevData {
    /**
     * @brief array of pointers to vertex data
     * @detail the first element points to the structure that contanins the
     *         edge pointer and the degree of the vertex
     */
    byte_t* d_vertex_ptrs[NUM_VTYPES];
    ///@brief number of vertices in the graph
    vid_t   nV;
};

/**
 * @brief Main cuStinger class
 */
class cuStinger {
public:
    /**
     * @brief default costructor
     * @param[in] custinger_init cuStinger initilialization data structure
     * @param[in] traspose if `true` traspose the input graph, keep the initial
     *            representation otherwise
     */
    explicit cuStinger(const cuStingerInit& custinger_init,
                       bool traspose = false) noexcept;

    /**
     * @brief Decostructor
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
     *          structure on the host and then perform the comparison
     */
    void check_consistency(const cuStingerInit& custinger_init) const noexcept;

    /**
     * @brief store the actual custinger representation to disk for future use
     * @param[in] filename name of the file where the graph
     */
    void store_snapshot(const std::string& filename) const noexcept;

    /**
     * @brief unique identifier of the cuStinger instance among all created
     *        instances
     * @return unique identifier
     */
    int id() const noexcept;

    /**
     * @brief **actual** number of vertices in the graph
     * @return actual number of vertices
     */
    size_t nV() const noexcept;

    /**
     * @brief **actual** number of edges in the graph
     * @return actual number of edges
     */
    size_t nE() const noexcept;

    /**
     * @brief **actual** csr offsets of the graph
     * @return pointer to csr offsets
     */
    const eoff_t* csr_offsets() const noexcept;

    /**
     * @brief **actual** csr edges of the graph
     * @return pointer to csr edges
     */
    const vid_t* csr_edges() const noexcept;

    /**
     * @brief **actual** device csr offsets of the graph
     * @return device pointer to csr offsets
     */
    const eoff_t* device_csr_offsets() noexcept;

    /**
     * @brief device data to used the cuStinger data structure on the device
     * @return device data associeted to the cuStinger instance
     */
    cuStingerDevData device_data() const noexcept;

    vid_t max_degree_vertex() const noexcept;

private:
    static int global_id;

    MemoryManager mem_manager;

    /**
     * @internal
     * @brief device pointer for *all* vertex data
     *        (degree and edge pointer included)
     */
    byte_t* _d_vertex_ptrs[NUM_VTYPES];

    const cuStingerInit& _custinger_init;
    const eoff_t* _csr_offsets;
    const vid_t*  _csr_edges;
    byte_t*       _d_vertices        { nullptr };
    eoff_t*       _d_csr_offsets     { nullptr };
    size_t        _nV;
    size_t        _nE;
    const int     _id;
    bool          _internal_csr_data { false };
    vid_t         _max_degree_vertex { -1 };

    void initialize() noexcept;

    /**
     * @internal
     * @brief traspose the cuStinger graph directly on the device
     */
    void transpose() noexcept;

    /**
     * @internal
     * @brief convert the actual cuStinger graph into csr offsets and csr edges
     * @param[out] csr_offsets csr offsets to build
     * @param[out] csr_offsets csr edges to build
     */
    void convert_to_csr(eoff_t* csr_offsets, vid_t* csr_edges) const noexcept;
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
     *            the source and the destination arrays (vid_t type)
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *config.inc* file
     * @see ::insertVertexData
     */
    template<typename... TArgs>
    void insertEdgeData(TArgs... edge_data) noexcept;

private:
    byte_t* _edge_data_ptrs[ NUM_ETYPES + 1 ]; //+1 for source ids
    size_t  _batch_size;
};

} // namespace custinger

#include "cuStinger.i.hpp"
