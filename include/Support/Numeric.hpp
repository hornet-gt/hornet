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
#include <algorithm>    //std::min

namespace xlib {

// nearest multiple of 4 : ((n-1)|3) + 1

// ==================== CONST EXPR TIME numeric methods ========================

template<typename T, typename... TArgs>
inline const T& min(const T& a, const TArgs&... args) noexcept {
    return std::min(a, xlib::min(args...));
}
template<typename T>
inline const T& min(const T& a, const T& b) noexcept {
    return std::min(a, b);
}

template<typename T>
HOST_DEVICE CONST_EXPR
bool addition_is_safe(T a, T b) noexcept;

template<typename T>
HOST_DEVICE CONST_EXPR
bool mul_is_safe(T a, T b) noexcept;

template<typename R, typename T>
void overflowT(T value);

template<typename T>
HOST_DEVICE CONST_EXPR
T ceil_div(T value, T div) noexcept;

template<typename T>
HOST_DEVICE CONST_EXPR
T uceil_div(T value, T div) noexcept;

template<uint64_t DIV, typename T>
HOST_DEVICE CONST_EXPR
T ceil_div(T value) noexcept;

/**
 *  15 / 4 = 3.75 --> 4
 *  13 / 4 = 3.25 --> 3
 * .0 to .0.4 round down, .5 to .9 round up
 */
template<typename T>
HOST_DEVICE CONST_EXPR
T round_div(T value, T div) noexcept;

template<typename T>
HOST_DEVICE CONST_EXPR
T uround_div(T value, T div) noexcept;

template<uint64_t DIV, typename T>
HOST_DEVICE CONST_EXPR
T round_div(T value) noexcept;

template<typename T>
HOST_DEVICE CONST_EXPR T upper_approx(T value, T mul) noexcept;

template<uint64_t MUL, typename T>
HOST_DEVICE CONST_EXPR T upper_approx(T value) noexcept;

template<typename T>
HOST_DEVICE CONST_EXPR T lower_approx(T value, T mul) noexcept;

template<int64_t MUL, typename T>
HOST_DEVICE CONST_EXPR T lower_approx(T value) noexcept;

template<typename T>
HOST_DEVICE CONST_EXPR bool is_power2(T value) noexcept;
template<typename T>
HOST_DEVICE CONST_EXPR T    factorial(T value) noexcept;

template<typename T, typename R>
HOST_DEVICE bool read_bit(const T* array, R pos) noexcept;

template<typename T, typename R>
HOST_DEVICE void write_bit(T* array, R pos) noexcept;

template<typename T, typename R>
HOST_DEVICE void write_bit(T* array, R start, R end) noexcept;

template<typename T, typename R>
HOST_DEVICE void delete_bit(T* array, R pos) noexcept;

template<typename T, typename R>
HOST_DEVICE void delete_bit(T* array, R start, R end) noexcept;

// ========================== RUN TIME numeric methods =========================

template<typename T>
HOST_DEVICE CONST_EXPR T roundup_pow2(T value) noexcept;

/** @fn T log2(const T value)
 *  @brief calculate the integer logarithm of 'value'
 *  @return &lfloor; log2 ( value ) &rfloor;
 */
template<typename T>
HOST_DEVICE int log2(T value) noexcept;

template<typename T>
HOST_DEVICE int ceil_log2(T value) noexcept;

template<typename T>
float per_cent(T part, T max) noexcept;

template<typename R>
struct CompareFloatABS {
    template<typename T>
    bool operator() (T a, T b) noexcept;
};

template<typename R>
struct CompareFloatRelativeErr {
    template<typename T>
    bool operator() (T a, T b) noexcept;
};

template <typename T, typename = void>
class WeightedRandomGenerator {
    template<typename R>
    WeightedRandomGenerator(const R* input, size_t size);

    size_t get() const noexcept;
};

} // namespace xlib

#include "impl/Numeric.i.hpp"
