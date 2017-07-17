/**
 * @brief cuStinger C-Style operators
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

#include "Core/cuStingerTypes.cuh"

namespace custinger_alg {
/////////////////
// C Style API //
/////////////////
/**
 * @brief Block size for all kernels associeted to operators
 */
const int BLOCK_SIZE_OP1 = 256;

/**
 * @brief apply the `Operator` to the algorithm-dependent data a fixed number of
 *        times
 * @tparam    Operator function to apply at each iteration to `optional_data`
 * @param[in] num_times number of iterations
 * @param[in] optional_data algorithm-dependent data
 */
template<void (*Operator)(int, void*)>
void forAll(int num_times, void* optional_data) noexcept;

/**
 * @brief apply the `Operator` to the algorithm-dependent data a number of times
 *        equal to the actual number of vertices in the graph
 * @tparam    Operator function to apply at each iteration to `optional_data`
 * @param[in] custinger custinger instance
 * @param[in] optional_data algorithm-dependent data
 */
template<void (*Operator)(custinger::vid_t, void*)>
void forAllnumV(const custinger::cuStinger& custinger, void* optional_data)
                noexcept;

/**
 * @brief apply the `Operator` to the algorithm-dependent data a number of times
 *        equal to the actual number of edges in the graph
 * @tparam    Operator function to apply at each iteration to `optional_data`
 * @param[in] custinger custinger instance
 * @param[in] optional_data algorithm-dependent data
 */
template<void (*Operator)(custinger::eoff_t, void*)>
void forAllnumE(const custinger::cuStinger& custinger, void* optional_data)
                noexcept;

//------------------------------------------------------------------------------

/**
 * @brief apply the `Operator` to the algorithm-dependent data for all vertices
 *        in the graph
 * @tparam    Operator function to apply at each iteration to a vertex and to
 *            the `optional_data`
 * @param[in] custinger custinger instance
 * @param[in] optional_data algorithm-dependent data
 */
template<void (*Operator)(custinger::Vertex&, void*)>
void forAllVertices(const custinger::cuStinger& custinger, void* optional_data)
                    noexcept;

template<void (*Operator)(custinger::Vertex&, void*)>
void forAllVertices(const custinger::cuStinger& custinger,
                    TwoLevelQueue<custinger::vid_t>& queue,
                    void* optional_data) noexcept;

/**
 * @brief apply the `Operator` to the algorithm-dependent data for all edges
 *        in the graph
 * @tparam    Operator function to apply at each iteration to an edge and to
 *            the `optional_data`
 * @param[in] custinger custinger instance
 * @param[in] optional_data algorithm-dependent data
 * @remark    the first call may be more expensive than the following
 */
template<void (*Operator)(custinger::Vertex&, const custinger::Edge&, void*)>
void forAllEdges(const custinger::cuStinger& custinger, void* optional_data)
                 noexcept;

template<void (*Operator)(custinger::Edge&, void*), typename LoadBalancing>
void forAllEdges(TwoLevelQueue<custinger::vid_t>& queue,
                 void* optional_data,
                 LoadBalancing& LB) noexcept;
//------------------------------------------------------------------------------
/*
template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllBatchEdges(const custinger::cuStinger& custinger,
                      const EdgeBatch& edge_batch, void* optional_data);

template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
template<typename Operator>
void forAllBatchVertices(Operator op);
*/
} // namespace custinger_alg

#include "Operator.i.cuh"
