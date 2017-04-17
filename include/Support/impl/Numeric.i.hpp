/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
 */
#include "Support/Basic.hpp"
#include "Support/Metaprogramming.hpp"
#if defined(__NVCC__)
    #include "Support/PTX.cuh"
#endif
#include <chrono>       //std::chrono
#include <cmath>        //std::abs
#include <limits>       //std::numeric_limits
#include <random>       //std::mt19937_64
#include <type_traits>  //std::is_integral

namespace xlib {

#if ((__clang_major__ >= 3 && __clang_minor__ >= 4) || __GNUC__ > 5)

template<typename T>
HOST_DEVICE CONST_EXPR
bool addition_is_safe(T a, T b) noexcept {
    return std::is_integral<T>::value ? __builtin_add_overflow(a, b) : true;
}

template<typename T>
HOST_DEVICE CONST_EXPR
bool mul_is_safe(T a, T b) noexcept {
    return std::is_integral<T>::value ? __builtin_mul_overflow(a, b) : true;
}

#else

template<typename T>
HOST_DEVICE CONST_EXPR
bool addition_is_safe(T a, T b) noexcept {
    return std::is_integral<T>::value && std::is_unsigned<T>::value ?
           (a + b) < a : true;
}

template<typename T>
HOST_DEVICE CONST_EXPR
bool mul_is_safe(T, T) noexcept { return true; }

#endif

template<typename R, typename T>
void overflowT(T value) {
    if (value > std::numeric_limits<R>::max())
        ERROR("value overflow")
}

namespace detail {

template<typename T>
HOST_DEVICE CONST_EXPR
typename std::enable_if<std::is_unsigned<T>::value, T>::type
ceil_div_aux(T value, T div) noexcept {
    //return value == 0 ? 0 : 1u + ((value - 1u) / div);       // not overflow
    return (value + div - 1) / div;       // may overflow
    //return (value / div) + ((value % div) > 0)
    //  --> remainer = zero op, but not GPU devices
}

template<typename T>
HOST_DEVICE CONST_EXPR
typename std::enable_if<std::is_signed<T>::value, T>::type
ceil_div_aux(T value, T div) noexcept {
    using R = typename std::make_unsigned<T>::type;
    return (value > 0) ^ (div > 0) ? value / div :
       static_cast<T>(ceil_div_aux(static_cast<R>(value), static_cast<R>(div)));
}

/**
 *
 * @warning division by zero
 * @warning division may overflow if (value + div / 2) > numeric_limits<T>::max()
 */
template<typename T>
HOST_DEVICE CONST_EXPR
typename std::enable_if<std::is_unsigned<T>::value, T>::type
round_div_aux(T value, T div) noexcept {
    CONST_EXPR_ASSERT(addition_is_safe(value, div / 2u) && "division overflow");
    return (value + (div / 2u)) / div;
}

/**
 *
 * @warning division by zero
 * @warning division may overflow/underflow. If value > 0 && div > 0 -> assert
 *         value / div > 0 --> (value - div / 2) may underflow
 *         value / div < 0 --> (value + div / 2) may overflow
 */
template<typename T>
HOST_DEVICE CONST_EXPR
typename std::enable_if<std::is_signed<T>::value, T>::type
round_div_aux(T value, T div) noexcept {
    CONST_EXPR_ASSERT(addition_is_safe(value, div / 2) && "division overflow");
    CONST_EXPR_ASSERT(value > 0 && div > 0 && "value, div > 0");
    return (value < 0) ^ (div < 0) ? (value - div / 2) / div
                                   : (value + div / 2) / div;
}

} // namespace detail

//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE CONST_EXPR
T ceil_div(T value, T div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    CONST_EXPR_ASSERT(div != 0 && "division by zero in integer arithmetic");

    return detail::ceil_div_aux(value, div);
}
template<typename T>
HOST_DEVICE CONST_EXPR
T uceil_div(T value, T div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    CONST_EXPR_ASSERT(div != 0 && "division by zero in integer arithmetic");

    using R = typename std::make_unsigned<T>::type;
    return detail::ceil_div_aux(static_cast<R>(value), static_cast<R>(div));
}

template<uint64_t DIV, typename T>
HOST_DEVICE CONST_EXPR
T ceil_div(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    static_assert(DIV != 0, "division by zero in integer arithmetic");
    CONST_EXPR_ASSERT(std::is_unsigned<T>::value || value >= 0);

    const auto DIV_ = static_cast<T>(DIV);
    //return value == 0 ? 0 : 1 + ((value - 1) / DIV_);
    return (value + DIV_ - 1) / DIV_;
}

//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE CONST_EXPR
T round_div(T value, T div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    CONST_EXPR_ASSERT(div != 0 && "division by zero in integer arithmetic");

    return detail::round_div_aux(value, div);
}
template<typename T>
HOST_DEVICE CONST_EXPR
T uround_div(T value, T div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    CONST_EXPR_ASSERT(div != 0 && "division by zero in integer arithmetic");

    using R = typename std::make_unsigned<T>::type;
    return detail::round_div_aux(static_cast<R>(value), static_cast<R>(div));
}

template<uint64_t DIV, typename T>
HOST_DEVICE CONST_EXPR
T round_div(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    static_assert(DIV > 0, "division by zero");
    CONST_EXPR_ASSERT(std::is_unsigned<T>::value || value >= 0);
    CONST_EXPR_ASSERT(addition_is_safe(value, static_cast<T>(DIV / 2u)));

    const auto DIV_ = static_cast<T>(DIV);
    return (value + (DIV_ / 2u)) / DIV_;
}

//------------------------------------------------------------------------------

/**
 * @pre T must be integral
 * @warning division by zero
 * @warning division may overflow if (value + (mul / 2)) > numeric_limits<T>::max()
 */
template<typename T>
HOST_DEVICE CONST_EXPR T upper_approx(T value, T mul) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return ceil_div(value, mul) * mul;
}

template<uint64_t MUL, typename T>
HOST_DEVICE CONST_EXPR T upper_approx(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    CONST_EXPR_ASSERT(std::is_unsigned<T>::value || value >= 0);

    const auto MUL_ = static_cast<T>(MUL);
    return MUL == 1 ? value :
            !IsPower2<MUL>::value ? ceil_div<MUL>(value) * MUL_
                                  : (value + MUL_ - 1) & ~(MUL_ - 1);
}

template<typename T>
HOST_DEVICE CONST_EXPR T lower_approx(T value, T mul) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return (value / mul) * mul;
}

template<int64_t MUL, typename T>
HOST_DEVICE CONST_EXPR T lower_approx(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    CONST_EXPR_ASSERT(std::is_unsigned<T>::value || value >= 0);

    const auto MUL_ = static_cast<T>(MUL);
    return MUL == 1 ? value :
           !IsPower2<MUL>::value ? (value / MUL_) * MUL_ : value & ~(MUL_ - 1);
}

//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE CONST_EXPR bool is_power2(T value) noexcept {
    using R = typename std::conditional<std::is_integral<T>::value,
                                        T, uint64_t>::type;
    auto value_ = static_cast<R>(value);
    CONST_EXPR_ASSERT(std::is_unsigned<R>::value || value_ >= 0);
    return (value_ != 0) && !(value_ & (value_ - 1));
}

template<typename T>
HOST_DEVICE CONST_EXPR T factorial(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return value <= 1 ? 1 : value * factorial(value - 1);
}

template<typename T, typename R>
HOST_DEVICE bool read_bit(const T* array, R pos) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto           upos = static_cast<unsigned>(pos);
    return array[upos / SIZE] & (static_cast<T>(1) << (upos % SIZE));
}

template<typename T, typename R>
HOST_DEVICE void write_bit(T* array, R pos) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto           upos = static_cast<unsigned>(pos);
    array[upos / SIZE] |= static_cast<T>(1) << (upos % SIZE);
}

template<typename T, typename R>
HOST_DEVICE void write_bit(T* array, R start, R end) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto         ustart = static_cast<unsigned>(start);
    auto           uend = static_cast<unsigned>(end);
    auto     start_word = ustart / SIZE;
    auto       end_word = uend / SIZE;
    array[start_word]  |= ~((static_cast<T>(1) << (uend % SIZE)) - 1);
    array[end_word]    |= (static_cast<T>(1) << (ustart % SIZE)) - 1;
    std::fill(array + start_word + 1, array + end_word - 1, static_cast<T>(-1));
}

template<typename T, typename R>
HOST_DEVICE void delete_bit(T* array, R pos) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto           upos = static_cast<unsigned>(pos);
    array[upos / SIZE] &= ~(static_cast<T>(1) << (upos % SIZE));
}

template<typename T, typename R>
HOST_DEVICE void delete_bit(T* array, R start, R end) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto         ustart = static_cast<unsigned>(start);
    auto           uend = static_cast<unsigned>(end);
    auto     start_word = ustart / SIZE;
    auto       end_word = uend / SIZE;
    array[start_word]  &= (static_cast<T>(1) << (ustart % SIZE)) - 1;
    array[end_word]    &= ~((static_cast<T>(1) << (uend % SIZE)) - 1);
    std::fill(array + start_word + 1, array + end_word - 1, 0);
}

// ========================== RUN TIME numeric methods =========================

template<typename T>
HOST_DEVICE CONST_EXPR T roundup_pow2(T value) noexcept {
    const bool is_integral = std::is_integral<T>::value;
    using R = typename std::conditional<is_integral, T, uint64_t>::type;
    auto  v = static_cast<R>(value);
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

/*
template<typename T>
HOST_DEVICE CONST_EXPR T rounddown_pow2(T value) noexcept {
    const bool is_integral = std::is_integral<T>::value;
    using R = typename std::conditional<is_integral, T, uint64_t>::type;
    R v = static_cast<R>(value);
    v |= (v >> 1);
    v |= (v >> 2);
    v |= (v >> 4);
    v |= (v >> 8);
    v |= (v >> 16);
    return(v & ~(v >> 1));
}*/

template<typename T>
HOST_DEVICE int log2(T value) noexcept {
    const bool is_integral = std::is_integral<T>::value;
    using R = typename std::conditional<is_integral && sizeof(T) <= 4, unsigned,
                                        uint64_t>::type;
    const auto value_unsigned = static_cast<R>(value);
    assert(value > 0);

    #if defined(__CUDA_ARCH__)
        return __msb(value_unsigned);
    #else
        return sizeof(T) < 8 ? 31 - __builtin_clz(value_unsigned) :
                               63 - __builtin_clzll(value_unsigned);
    #endif
}

template<typename T>
HOST_DEVICE int ceil_log2(T value) noexcept {
    return is_power2(value) ? log2(value) : log2(value) + 1;
}

template<unsigned BASE, typename T>
HOST_DEVICE typename std::enable_if<!xlib::IsPower2<BASE>::value, int>::type
log(T value) {
    int count;
    for (count = 0; value; count++)
        value /= BASE;
    return count;
}

template<unsigned BASE, typename T>
HOST_DEVICE typename std::enable_if<xlib::IsPower2<BASE>::value, int>::type
log(T value) {
    return xlib::log2(value) / xlib::Log2<BASE>::value;
}

template<unsigned BASE, typename T>
HOST_DEVICE typename std::enable_if<!xlib::IsPower2<BASE>::value, int>::type
ceil_log(T value);

template<unsigned BASE, typename T>
HOST_DEVICE typename std::enable_if<xlib::IsPower2<BASE>::value, int>::type
ceil_log(T value) {
    return xlib::ceil_log2(value) / xlib::Log2<BASE>::value;
}

template<unsigned BASE, typename T>
HOST_DEVICE typename std::enable_if<xlib::IsPower2<BASE>::value, T>::type
pow(T value) {
    return 1 << (xlib::Log2<BASE>::value * value);
}

template<typename T>
float per_cent(T part, T max) noexcept {
    return (static_cast<float>(part) / static_cast<float>(max)) * 100.0f;
}

HOST_DEVICE
unsigned multiplyShiftHash32(unsigned A, unsigned B, unsigned log_bins,
                             unsigned value) noexcept {
    return static_cast<unsigned>(A * value + B) >> (32 - log_bins);
}
HOST_DEVICE
uint64_t multiplyShiftHash64(uint64_t A, uint64_t B, unsigned log_bins,
                             uint64_t value) noexcept {
    return static_cast<uint64_t>(A * value + B) >> (64 - log_bins);
}

template<unsigned A, unsigned B, unsigned BINS>
struct MultiplyShiftHash32 {
    static_assert(IsPower2<BINS>::value, "BINS must be a power of 2");

    HOST_DEVICE
    static unsigned op(unsigned value) {
        return static_cast<unsigned>(A * value + B) >> (32 - Log2<BINS>::value);
    }
};

template<uint64_t A, uint64_t B, unsigned BINS>
struct MultiplyShiftHash64 {
    static_assert(IsPower2<BINS>::value, "BINS must be a power of 2");

    HOST_DEVICE
    static uint64_t op(uint64_t value) {
        return static_cast<uint64_t>(A * value + B) >> (64 - Log2<BINS>::value);
    }
};

template<std::intmax_t Num, std::intmax_t Den>
struct CompareFloatABS<std::ratio<Num, Den>> {
    template<typename T>
    inline bool operator() (T a, T b) noexcept {
        const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
        return std::abs(a - b) < epsilon;
    }
};

template<std::intmax_t Num, std::intmax_t Den>
struct CompareFloatRelativeErr<std::ratio<Num, Den>> {
    template<typename T>
    inline bool operator() (T a, T b) noexcept {
        const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
        const T diff = std::abs(a - b);
        //return (diff < epsilon) ||
        //       (diff / std::max(std::abs(a), std::abs(b)) < epsilon);
        return (diff < epsilon) ||
               ( diff / std::min(std::abs(a) + std::abs(b),
                                 std::numeric_limits<float>::max()) < epsilon);
    }
};

//------------------------------------------------------------------------------

namespace detail {

template <typename T>
class WeightedRandomGeneratorAux {
public:
    template<typename R>
    WeightedRandomGeneratorAux(const R* weights, size_t size) :
            _cumulative(nullptr), _size(size),
            _gen(static_cast<uint64_t>(
                 std::chrono::system_clock::now().time_since_epoch().count())) {

        _cumulative    = new T[size + 1];
        _cumulative[0] = 0;
        for (size_t i = 1; i <= size; i++)
            _cumulative[i] = _cumulative[i - 1]+ static_cast<T>(weights[i - 1]);
    }
    ~WeightedRandomGeneratorAux() {
        delete[] _cumulative;
    }
protected:
    std::mt19937_64 _gen;
    T*              _cumulative;
    size_t          _size;
};

} //namespace detail

template <typename T>
class WeightedRandomGenerator<T, typename std::enable_if<
                                    std::is_integral<T>::value>::type> :
                                public detail::WeightedRandomGeneratorAux<T> {
public:
    using detail::WeightedRandomGeneratorAux<T>::_cumulative;
    using detail::WeightedRandomGeneratorAux<T>::_gen;
    using detail::WeightedRandomGeneratorAux<T>::_size;

    template<typename R>
    WeightedRandomGenerator(const R* weights, size_t size) :
            detail::WeightedRandomGeneratorAux<T>(weights, size), _int_distr() {

        using param_t = typename std::uniform_int_distribution<T>::param_type;
        _int_distr.param(param_t(0, _cumulative[size] - 1));
    }

    inline size_t get() noexcept {
        T value = _int_distr(_gen);
        auto it = std::upper_bound(_cumulative, _cumulative + _size, value);
        return std::distance(_cumulative, it) - 1;
    }
private:
    std::uniform_int_distribution<T> _int_distr;
};

template <typename T>
class WeightedRandomGenerator<T, typename std::enable_if<
                                    std::is_floating_point<T>::value>::type> :
                                public detail::WeightedRandomGeneratorAux<T> {
public:
    using detail::WeightedRandomGeneratorAux<T>::_cumulative;
    using detail::WeightedRandomGeneratorAux<T>::_gen;
    using detail::WeightedRandomGeneratorAux<T>::_size;

    template<typename R>
    WeightedRandomGenerator(const R* weights, size_t size) :
           detail::WeightedRandomGeneratorAux<T>(weights, size), _real_distr() {

        using param_t = typename std::uniform_real_distribution<T>::param_type;
        _real_distr.param(param_t(0, _cumulative[size - 1]));
    }

    inline size_t get() noexcept {
        T value = _real_distr(_gen);
        auto it = std::upper_bound(_cumulative, _cumulative + _size, value);
        return std::distance(_cumulative, it);
    }
private:
    std::uniform_real_distribution<T> _real_distr;
};

} // namespace xlib
