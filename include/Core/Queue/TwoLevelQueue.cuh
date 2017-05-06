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

#include <Core/cuStinger.hpp>

namespace custinger_alg {

template<typename T>
struct ptr2_t {
    const T* first;
    T*       second;
    void swap() noexcept;
};

template<typename T>
class TwoLevelQueue {
static const bool is_vid = std::is_same<T, custinger::vid_t>::value;
using     EnableTraverse = typename std::enable_if< is_vid >::type;
public:
    explicit TwoLevelQueue(size_t max_allocated_items) noexcept;
    explicit TwoLevelQueue(size_t max_allocated_items,
                           const custinger::eoff_t* csr_offset) noexcept;

    ~TwoLevelQueue() noexcept;

    __host__ __device__ void insert(const T& item) noexcept;

    __host__ void insert(const T* items_array, int num_items) noexcept;

    __host__ void swap() noexcept;

    __host__ void clear() noexcept;

    __host__ int size() const noexcept;

    __host__ const T* device_ptr_q1() const noexcept;

    __host__ const T* device_ptr_q2() const noexcept;

    __host__ const T* host_data() noexcept;

    __host__ void print() const noexcept;

    template<typename Operator>
    __host__ EnableTraverse
    traverse_edges(Operator op) noexcept;

private:
    static const bool     CHECK_CUDA_ERROR1 = true;
    static const bool PRINT_VERTEX_FRONTIER = 0;
    static const unsigned        BLOCK_SIZE = 256;

    const custinger::eoff_t* _csr_offsets { nullptr };

    ptr2_t<T>   _d_queue_ptrs        { nullptr, nullptr };
    ptr2_t<int> _d_work_ptrs         { nullptr, nullptr };
    int*        _d_queue_counter     { nullptr };
    T*          _host_data           { nullptr };
    size_t      _max_allocated_items;
    int         _num_queue_vertices  { 0 };
    int         _num_queue_edges     { 0 };   // traverse_edges
    bool        _enable_traverse     { false };

    __host__ void work_evaluate(const T* items_array, int num_items) noexcept;
};

} // namespace custinger_alg

#include "TwoLevelQueue.i.cuh"
