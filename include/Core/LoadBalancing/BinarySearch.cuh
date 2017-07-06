/**
 * @internal
 * @brief Device-wide Binary Search load balacing
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
#include "Core/Queue/TwoLevelQueue.cuh"

/**
 * @brief The namespace provides all load balacing methods to traverse vertices
 */
namespace load_balacing {

/**
 * @brief The class implements the BinarySearch load balacing
 */
class BinarySearch {
public:
    /**
     * @brief Default costructor
     * @param[in] custinger cuStinger instance
     */
    //explicit BinarySearch(const custinger::cuStinger& custinger) noexcept;

    /**
     * @brief Decostructor
     */
    //~BinarySearch() noexcept;

    /**
     * @brief Traverse the edges in a vertex queue (C-Style API)
     * @tparam Operator function to apply to each edge and to `optional_data`
     * @param[in] queue input vertex queue
     * @param[in] optional_field algorithm-dependent data
     */
    /*template<void (*Operator)(const custinger::Vertex&, const custinger::Edge&,
                              void*)>
    void traverse_edges(const
                        custinger_alg::TwoLevelQueue<custinger::vid_t>& queue,
                        void* optional_field) noexcept;*/

    /**
     * @brief Traverse the edges in a device vertex array (C-Style API)
     * @tparam Operator function to apply to each edge and to `optional_data`
     * @param[in] d_input device vertex array
     * @param[in] num_vertices number of vertices in the input array
     * @param[in] optional_field algorithm-dependent data
     */
    /*template<void (*Operator)(const custinger::Vertex&, const custinger::Edge&,
                              void*)>
    void traverse_edges(const custinger::vid_t* d_input, int num_vertices,
                        void* optional_field) noexcept;*/

    /**
     * @brief Traverse the edges in a vertex queue (C++11-Style API)
     * @tparam Operator function to apply at each edge
     * @param[in] queue input vertex queue
     * @param[in] op struct/lambda expression that implements the operator
     * @remark    all algorithm-dependent data must be capture by `op`
     * @remark    the Operator typename must implement the method
     *            `void operator()(Vertex, Edge)` or the lambda expression
     *            `[=](Vertex, Edge){}`
     */
     template<typename Operator>
     void apply(custinger::cuStinger& custinger,
                const custinger::vid_t* d_input, int num_vertices,
                const Operator& op) noexcept;

    template<typename Operator>
    void apply(custinger::cuStinger& custinger, const Operator& op) noexcept;

    /**
     * @brief Traverse the edges in a vertex array (C++11-Style API)
     * @tparam Operator function to apply at each edge
     * @param[in] d_input vertex array
     * @param[in] num_vertices number of vertices in the queue
     * @param[in] op struct/lambda expression that implements the operator
     * @remark    all algorithm-dependent data must be capture by `op`
     * @remark    the Operator typename must implement the method
     *            `void operator()(Vertex, Edge)` or the lambda expression
     *            `[=](Vertex, Edge){}`
     */
    //template<typename Operator>
    //void traverse_edges(const custinger::vid_t* d_input, int num_vertices,
    //                    Operator op) noexcept;

private:
    static const int         BLOCK_SIZE = 256;
    static const bool CHECK_CUDA_ERROR1 = 1;
    //const custinger::cuStinger& _custinger;
    int* _d_work    { nullptr };
    int* _d_degrees { nullptr };
};

} // namespace load_balacing

#include "Core/LoadBalancing/BinarySearch.i.cuh"
