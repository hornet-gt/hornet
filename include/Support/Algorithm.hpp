/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Support/HostDevice.hpp"
#include <unordered_map>    //std::unordered_map

namespace xlib {

template<class FUN_T, typename... T>
inline void Funtion_TO_multiThreads(bool MultiCore, FUN_T FUN, T... Args);

/**
 * @brief return the old value if exits
 */
template<typename T, typename R = T>
class UniqueMap : public std::unordered_map<T, R> {
static_assert(std::is_integral<R>::value,
              "UniqueMap accept only Integral types");
public:
    R insertValue(T id);
};

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
        bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type));

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equalSorted(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

/**
 * required auxilary space: O(|end -start| * 2)
 */
template<typename T, typename... RArgs>
void sort_by_key(T* start, T* end, RArgs... data_packed);

//==============================================================================

template<typename T, typename R>
HOST_DEVICE
R lower_bound_left(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R lower_bound_right(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R upper_bound_left(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R upper_bound_right(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R binary_search(const T* mem, R size, T searched);

//------------------------------------------------------------------------------

template<typename T, typename S>
HOST_DEVICE
void merge(const T* left, S size_left, const T* right, S size_right, T* merge);

template<typename T, typename S>
HOST_DEVICE
void inplace_merge(T* left, S size_left, const T* right, S size_right);

} // namespace xlib

#include "impl/Algorithm.i.hpp"
