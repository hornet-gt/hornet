/**
 * @brief Hornet C++11 operators
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
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

#include "Queue/TwoLevelQueue.cuh"
#include "HostDeviceVar.cuh"
#include "LoadBalancing/VertexBased.cuh"
#include <BasicTypes.hpp>
//#include <Core/BatchUpdate/BatchUpdate.cuh>

namespace hornets_nest {

///////////////
// C++11 API //
///////////////
/**
 * @brief Block size for all kernels associated to operators
 */
const int BLOCK_SIZE_OP2 = 256;

#define OPERATOR template<typename Vertex = void, typename Edge = void>        \
                 __device__ __forceinline__                                    \
                 void operator()

/**
 * @brief apply the `Operator` a fixed number of times
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] num_times number of iterations
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 */
template<typename Operator>
void forAll(int num_items, const Operator& op);

template<typename T, typename Operator>
void forAll(const TwoLevelQueue<T>& queue,
            const Operator&         op);

template<typename HornetClass, typename T, typename Operator>
void forAllVertexPairs(HornetClass&            hornet,
                       const TwoLevelQueue<T>& queue,
                       const Operator&         op);

/**
 * @brief apply the `Operator` a number of times equal to the actual number of
 *        vertices in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger Hornet instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 */
template<typename HornetClass, typename Operator>
void forAllnumV(HornetClass& hornet, const Operator& op);


/**
 * @brief apply the `Operator` a number of times equal to the actual number of
 *        edges in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger Hornet instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 */
template<typename HornetClass, typename Operator>
void forAllnumE(HornetClass& hornet, const Operator& op);

//==============================================================================
//==============================================================================

/**
 * @brief apply the `Operator` to all vertices in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger Hornet instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 * @remark    the Operator typename must implement the method
 *            `void operator()(Vertex)` or the lambda expression
 *            `[=](Vertex){}`
 */
template<typename HornetClass, typename Operator>
void forAllVertices(HornetClass& hornet, const Operator& op);

/**
 * @brief apply the `Operator` to all edges in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger Hornet instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 * @remark    the Operator typename must implement the method
 *            `void operator()(Vertex, Edge)` or the lambda expression
 *            `[=](Vertex, Edge){}`
 */
template<typename HornetClass, typename Operator>
void forAllEdges(HornetClass& hornet, const Operator& op);

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&         hornet,
                 const Operator&      op,
                 const LoadBalancing& load_balancing);

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdgeVertexPairs(HornetClass&         hornet,
                           const Operator&      op,
                           const LoadBalancing& load_balancing);

//==============================================================================
//==============================================================================

template<typename HornetClass, typename Operator>
void forAllVertices(HornetClass&    hornet,
                    const typename HornetClass::VertexType *    vertex_array,
                    int             size,
                    const Operator& op);

template<typename HornetClass, typename Operator>
void forAllVertices(HornetClass&                hornet,
                    const TwoLevelQueue<typename HornetClass::VertexType >& queue,
                    const Operator&             op);

//------------------------------------------------------------------------------

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&         hornet,
                 const typename HornetClass::VertexType *         vertex_array,
                 int                  size,
                 const Operator&      op);

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&         hornet,
                 const typename HornetClass::VertexType *         vertex_array,
                 int                  size,
                 const Operator&      op,
                 const LoadBalancing& load_balancing);

template<typename HornetClass, typename Operator>
void forAllAdjUnions(HornetClass&         hornet,
                     const Operator&      op,
                     const int WORK_FACTOR=1);

template<typename HornetClass, typename Operator>
void forAllAdjUnions(HornetClass&         hornet,
                     TwoLevelQueue<vid2_t> vertex_pairs,
                     const Operator&      op,
                     const int WORK_FACTOR=1);

template<typename HornetClass, typename Operator>
void forAllEdgesAdjUnionSequential(HornetClass &hornet, typename HornetClass::VertexType * queue, const unsigned long long size, const Operator &op, int flag);
template<typename HornetClass, typename Operator>
void forAllEdgesAdjUnionBalanced(HornetClass &hornet, typename HornetClass::VertexType * queue, const unsigned long long start, const unsigned long long end, const Operator &op, unsigned long long threads_per_union, int flag);

template<typename HornetClass, typename Operator>
void forAllEdgesAdjUnionImbalanced(HornetClass &hornet, typename HornetClass::VertexType * queue, const unsigned long long start, const unsigned long long end, const Operator &op, unsigned long long threads_per_union, int flag);

/**
 * @brief apply the `Operator` to all vertices in the graph
 * @tparam    Operator typename of the operator (deduced)
 * @param[in] custinger Hornet instance
 * @param[in] op struct/lambda expression that implements the operator
 * @remark    all algorithm-dependent data must be capture by `op`
 * @remark    the Operator typename must implement the method
 *            `void operator()(Vertex)` or the lambda expression
 *            `[=](Vertex){}`
 */
template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&                hornet,
                 const TwoLevelQueue<typename HornetClass::VertexType >& queue,
                 const Operator&             op);
/*
template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&                hornet,
                 const TwoLevelQueue<HornetClass::VertexType >& queue,
                 const Operator&             op,
                 const LoadBalancing&        load_balancing);
*/
template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&                hornet,
                 const TwoLevelQueue<typename HornetClass::VertexType >& queue,
                 const Operator&             op,
                 const LoadBalancing&        load_balancing);

//==============================================================================

//using BatchUpdate = hornet::BatchUpdate;
//
//template<typename HornetClass, typename Operator>
//void forAllEdges(HornetClass& hornet,
//                 const BatchUpdate& batch_update,
//                 const Operator& op);
//
//template<typename HornetClass, typename Operator>
//void forAllVertices(HornetClass& hornet,
//                    const BatchUpdate& batch_update,
//                    const Operator& op);

} // namespace hornets_nest

#include "Operator++.i.cuh"
