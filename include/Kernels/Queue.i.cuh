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

#include "Core/cuStingerGlobalSpace.hpp"

/**
 * @brief
 */
namespace cu_stinger_alg {

template<typename T>
Queue<T>::Queue() noexcept {
    size_t nV;
    cuMemcpyFromSymbol(d_nV, nV);

    cuMalloc(d_queue, nV * ALLOCATION_FACTOR);
    cuMemcpyToSymbol(d_queue, d_queue1);

    cuMalloc(d_queue_aux, nV * ALLOCATION_FACTOR);
    cuMemcpyToSymbol(d_queue_aux, d_queue2);

    cuMalloc(d_work_ptr, nV * ALLOCATION_FACTOR);
    cuMemcpyToSymbol(d_work_ptr, d_work);

    cuMemcpyToSymbol(0, d_queue_counter);
}

template<typename T>
~Queue() noexcept {
    cuFree(d_queue, d_queue_aux, d_work_ptr);
}

template<typename T>
__host__ void Queue<T>::insert(T item) noexcept {
    cuMemcpyToDeviceAsync(item, d_queue);
    _size++;
}

template<typename T>
__host__ void Queue<T>::insert(const T* item_array, int size) noexcept {
    cuMemcpyToDeviceAsync(item_array, size, d_queue);
    _size += size
}

template<typename T>
__host__ int Queue<T>::size() const noexcept {
    cuMemcpyToHostAsync(d_queue_counter, _size);
    return _size;
}

} // namespace cu_stinger_alg
