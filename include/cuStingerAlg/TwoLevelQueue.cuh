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

#include "Support/Device/VectorUtil.cuh"

namespace cu_stinger_alg {

template<typename T>
struct ptr2_t {
    T* first;
    T* second;

    void swap() noexcept;
};

__device__   int          d_queue_counter;
__constant__ ptr2_t<void> d_queue_ptrs;

template<typename T>
class TwoLevelQueue {
public:
    explicit TwoLevelQueue(size_t max_allocated_items) noexcept;
    ~TwoLevelQueue() noexcept;

    __host__ void insert(const T& item) noexcept;

    __host__ void insert(const T* items_array, int num_items) noexcept;

    __host__ int size() const noexcept;

    //__host__ int update_size() noexcept;

    __host__ void update_size(int size) noexcept;

    __host__ void swap() noexcept;

    __host__ const T* host_data() noexcept;

    __host__ void print() const noexcept;

    __host__ int max_allocated_items() const noexcept;
private:
    ptr2_t<T> _d_queue             { nullptr, nullptr };
    int*      _d_queue_counter     { nullptr };
    T*        _host_data           { nullptr };
    size_t    _max_allocated_items { 0 };
    int       _size                { 0 };
};

} // namespace cu_stinger_alg

#include "TwoLevelQueue.i.cuh"
