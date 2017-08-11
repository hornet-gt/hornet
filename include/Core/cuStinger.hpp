/**
 * @brief cuStinger, cuStingerInit, BatchUpdatem and BatchProperty classes
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
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

#include "Core/cuStingerDevice.cuh"     //cuStingerDevice
#include "Core/cuStingerInit.hpp"       //cuStingerInit
#include "Core/BatchUpdate.cuh"         //BatchUpdate
#include "Core/MemoryManager/MemoryManager.hpp"
#include "Core/RawTypes.hpp"
#include <cstddef>                      //size_t

/**
 * @brief The namespace contanins all classes and methods related to the
 *        cuStinger data structure
 */
namespace custinger {

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


    void check_sorted_adjs() const noexcept;

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
    const eoff_t* csr_offsets() noexcept;

    /**
     * @brief **actual** csr edges of the graph
     * @return pointer to csr edges
     */
    const vid_t* csr_edges() noexcept;

    /**
     * @brief **actual** device csr offsets of the graph
     * @return device pointer to csr offsets
     */
    const eoff_t* device_csr_offsets() noexcept;

    /**
     * @brief device data to used the cuStinger data structure on the device
     * @return device data associeted to the cuStinger instance
     */
    cuStingerDevice device_side() const noexcept;

    vid_t max_degree_id() noexcept;

    degree_t max_degree() noexcept;

    //--------------------------------------------------------------------------

    //void allocateBatch(const BatchProperty& batch_prop,
    //                   size_t max_allocated_edges = 0) noexcept;

    //void copyBatchToDevice(BatchHost& batch_host) noexcept;

    //void insertEdgeBatch(BatchUpdate& batch_update) noexcept;

    /*template<typename EqualOp>
    void insertEdgeBatch(BatchUpdate& batch_update, const EqualOp& equal_op)
                         noexcept;

    void edgeDeletionsSorted(BatchUpdate& batch_update) noexcept;*/

    //--------------------------------------------------------------------------
    void allocateEdgeDeletion(vid_t max_batch_size,
                              BatchProperty batch_prop) noexcept;

    void allocateEdgeInsertion(vid_t max_batch_size,
                               BatchProperty batch_prop) noexcept;

    void insertEdgeBatch(BatchUpdate& batch_update,
                         BatchProperty batch_prop) noexcept;

    template<typename EqualOp>
    void insertEdgeBatch(BatchUpdate& batch_update, const EqualOp& equal_op,
                         BatchProperty batch_prop = BatchProperty()) noexcept;

    void deleteEdgeBatch(BatchUpdate& batch_update,
                         BatchProperty batch_prop = BatchProperty()) noexcept;

private:
    static int global_id;

    MemoryManager mem_manager;

    /**
     * @internal
     * @brief device pointer for *all* vertex data
     *        (degree and edge pointer included)
     */
    byte_t* _d_vertex_ptrs[NUM_VTYPES] = {};
    byte_t* _d_edge_ptrs[NUM_ETYPES]   = {};

    const cuStingerInit& _custinger_init;
    const eoff_t* _csr_offsets       { nullptr };
    const vid_t*  _csr_edges         { nullptr };
    byte_t*       _d_vertices        { nullptr };
    byte_t*       _d_edges           { nullptr };   //for CSR
    degree_t*     _d_degrees         { nullptr };
    eoff_t*       _d_csr_offsets     { nullptr };
    size_t        _nV                { 0 };
    size_t        _nE                { 0 };
    const int     _id                { 0 };
    bool          _internal_csr_data { false };
    bool          _is_sorted         { false };

    std::pair<degree_t, vid_t> max_degree_data { -1, -1 };

    //----------------------------------------------------------------------
    BatchProperty _batch_prop;
    ///Batch delete tmp variables
    vid_t*    _d_unique       { nullptr };
    int*      _d_counts       { nullptr };
    degree_t* _d_degree_old   { nullptr };
    degree_t* _d_degree_new   { nullptr };
    byte_t*   *_d_ptrs_array  { nullptr };
    edge_t*   _d_tmp          { nullptr };
    bool*     _d_flags        { nullptr };
    eoff_t*   _d_inverse_pos  { nullptr };
    vid_t*    _d_tmp_sort_src { nullptr };
    vid_t*    _d_tmp_sort_dst { nullptr };
    size_t    _batch_pitch    { 0 };

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
    void convert_to_csr(eoff_t* csr_offsets, vid_t* csr_edges)
                        const noexcept;

    void build_batch_csr(int num_uniques);

    void send_to_device(BatchUpdate& batch_udate, BatchProperty batch_prop)
                        noexcept;

    void build_device_degrees() noexcept;
};

} // namespace custinger

#include "impl/cuStinger.i.hpp"
