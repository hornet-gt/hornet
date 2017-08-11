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

#include "Csr/RawTypes.hpp"
#include <cstddef>                  //size_t
#include "Core/cuStingerInit.hpp"   //cuStingerInit

/**
 * @brief
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
     */
    explicit cuStinger(const custinger::cuStingerInit& custinger_init) noexcept;

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

    vid_t max_degree_id() const noexcept;

    degree_t max_degree() const noexcept;

private:
    const cuStingerInit& _custinger_init;
    /**
     * @internal
     * @brief device pointer for *all* vertex data
     *        (degree and edge pointer included)
     */
    byte_t* _d_vertex_ptrs[NUM_VTYPES] = {};

    /**
     * @internal
     * @brief device pointer for *all* vertex data
     *        (degree and edge pointer included)
     */
    byte_t*      _d_vertices    { nullptr };
    byte_t*      _d_edges       { nullptr };
    eoff_t*      _d_csr_offsets { nullptr };
    size_t       _nV;
    size_t       _nE;

    void initialize() noexcept;
};

} // namespace custinger

#include "Core/impl/cuStinger.i.hpp"
