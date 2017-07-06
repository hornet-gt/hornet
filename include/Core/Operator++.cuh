/**
 * @brief cuStinger C++11 operators
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

namespace custinger_alg {
///////////////
// C++11 API //
///////////////
/**
 * @brief Block size for all kernels associeted to operators
 */
const int BLOCK_SIZE_OP2 = 256;

//enum class LoadBalancing { SIMPLE, BINARY_SEARCH, NODE_BASED, SCAN_BASED };

/**
 * @brief apply the `Operator` a fixed number of times
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] num_times number of iterations
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 */
template<typename Operator>
void forAll(int num_items, const Operator& op);

/**
 * @brief apply the `Operator` a number of times equal to the actual number of
 *        vertices in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger cuStinger instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 */
template<typename Operator>
void forAllnumV(const custinger::cuStinger& custinger, const Operator& op);


/**
 * @brief apply the `Operator` a number of times equal to the actual number of
 *        edges in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger cuStinger instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 */
template<typename Operator>
void forAllnumE(const custinger::cuStinger& custinger, const Operator& op);

//------------------------------------------------------------------------------

/**
 * @brief apply the `Operator` to all vertices in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger cuStinger instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 * @remark    the Operator typename must implement the method
 *            `void operator()(Vertex)` or the lambda expression
 *            `[=](Vertex){}`
 */
template<typename Operator>
void forAllVertices(custinger::cuStinger& custinger, const Operator& op);

/**
 * @brief apply the `Operator` to all edges in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger cuStinger instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 * @remark    the Operator typename must implement the method
 *            `void operator()(Vertex, Edge)` or the lambda expression
 *            `[=](Vertex, Edge){}`
 */
template<typename Operator, typename LoadBalancing>
void forAllEdges(custinger::cuStinger& custinger, const Operator& op,
                 LoadBalancing& LB);

//------------------------------------------------------------------------------

template<typename Operator>
void forAllVertices(custinger::cuStinger& custinger,
                    TwoLevelQueue<custinger::vid_t>& queue,
                    const Operator& op);

/**
 * @brief apply the `Operator` to all vertices in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger cuStinger instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 * @remark    the Operator typename must implement the method
 *            `void operator()(Vertex)` or the lambda expression
 *            `[=](Vertex){}`
 */
template<typename Operator, typename LoadBalancing>
void forAllEdges(custinger::cuStinger& custinger,
                 TwoLevelQueue<custinger::vid_t>& queue,
                 const Operator& op, LoadBalancing& LB);

//------------------------------------------------------------------------------

template<typename Operator>
void forAllBatchEdges(const Operator& op);

template<typename Operator>
void forAllBatchVertices(const Operator& op);

} // namespace custinger_alg

#include "Operator++.i.cuh"
