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

#include "Core/Queue/TwoLevelQueue.cuh"
#include <vector>

namespace custinger_alg {

/**
 * @warning known limitations: only one instance if allowed
 */
template<typename T>
class MultiLevelQueue {
static const bool is_vid = std::is_same<T, custinger::vid_t>::value;
using     EnableTraverse = typename std::enable_if< is_vid >::type;
public:
    explicit MultiLevelQueue(size_t max_allocated_items) noexcept;
    ~MultiLevelQueue() noexcept;

    __host__ __device__ void insert(const T& item) noexcept;

    __host__ void insert(const T* items_array, int num_items) noexcept;

    __host__ void next() noexcept;


    __host__ int size() const noexcept;

    __host__ int size(int level) const noexcept;

    __host__ const T* device_ptr() const noexcept;

    __host__ const T* device_ptr(int level) const noexcept;

    __host__ const T* host_data() noexcept;

    __host__ const T* host_data(int level) noexcept;

    __host__ void print() const noexcept;

    __host__ void print(int level) const noexcept;

    template<typename Operator>
    __host__ EnableTraverse
    work_evaluate(const custinger::eoff_t* csr_offsets) noexcept;

    template<typename Operator>
    __host__ EnableTraverse
    traverse_edges(Operator op) noexcept;

private:
    std::vector<int> _level_sizes;
    ptr2_t<T>        _d_queue_ptrs  { nullptr, nullptr };
    T*               _d_multiqueue  { nullptr };
    T*               _host_data     { nullptr };
    size_t           _max_allocated_items;
    int              _current_level { 0 };
};

} // namespace custinger_alg

#include "MultiLevelQueue.i.cuh"
