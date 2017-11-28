/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date October, 2017
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
#include "Device/Util/Atomic.cuh"
#include "Device/Util/Basic.cuh"
#include "Device/Util/PTX.cuh"

namespace xlib {
namespace detail {

#define WARP_SEG_REDUCE_MACRO(ASM_OP, ASM_T, ASM_CL)                           \
    const unsigned member_mask = xlib::member_mask<WARP_SZ>();                 \
    const unsigned max_lane    = segmented_maxlane<WARP_SZ>(mask);             \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{"                                                                \
            ".reg ."#ASM_T" r1;"                                               \
            ".reg .pred p;"                                                    \
            "shfl.sync.down.b32 r1|p, %1, %2, %3, %4;"                         \
            "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                \
            "mov.b32 %0, r1;"                                                  \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(1 << STEP),              \
              "r"(max_lane), "r"(member_mask));                                \
    }

#define WARP_SEG_REDUCE_MACRO2(ASM_OP, ASM_T, ASM_CL)                          \
    const unsigned member_mask = xlib::member_mask<WARP_SZ>();                 \
    const unsigned    max_lane = segmented_maxlane<WARP_SZ>(mask);             \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{"                                                                \
            ".reg .u32 lo;"                                                    \
            ".reg .u32 hi;"                                                    \
            ".reg .pred p;"                                                    \
            "mov.b64 {lo, hi}, %1;"                                            \
            "shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"                         \
            "shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"                         \
            "mov.b64 %0, {lo, hi};"                                            \
            "@p "#ASM_OP"."#ASM_T" %0, %0, %1;"                                \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(1 << STEP),              \
              "r"(max_lane), "r"(member_mask));                                \
    }

constexpr unsigned vwarp_mask(int WARP_SZ) {
    unsigned value = 0;
    for (int i = 0; i < 32; i += WARP_SZ) {
        value <<= WARP_SZ;
        value |= 1;
    }
    return value;
}

template<unsigned WARP_SZ>
__device__ __forceinline__
unsigned segmented_maxlane(unsigned mask) {
    return xlib::WARP_SIZE == WARP_SZ ? xlib::max_lane(mask) :
                xlib::max_lane(mask | vwarp_mask(WARP_SZ));
}

template<unsigned WARP_SZ>
__device__ __forceinline__
unsigned segmented_minlane(unsigned mask) {
    return xlib::WARP_SIZE == WARP_SZ ? xlib::min_lane(mask) :
                xlib::min_lane(mask | vwarp_mask(WARP_SZ));
}
//==============================================================================
//==============================================================================

template<int WARP_SZ, typename T>
struct WarpSegReduceHelper;

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, int> {

    __device__ __forceinline__
    static void add(int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(add, s32, r)
    }

    __device__ __forceinline__
    static void min(int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(min, s32, r)
    }

    __device__ __forceinline__
    static void max(int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(max, s32, r)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, unsigned> {

    __device__ __forceinline__
    static void add(unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(add, u32, r)
    }

    __device__ __forceinline__
    static void min(unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(min, u32, r)
    }

    __device__ __forceinline__
    static void max(unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(max, u32, r)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, float> {

    __device__ __forceinline__
    static void add(float& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(add, f32, f)
    }

    __device__ __forceinline__
    static void min(float& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(min, f32, f)
    }

    __device__ __forceinline__
    static void max(float& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO(max, f32, f)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, double> {

    __device__ __forceinline__
    static void add(double& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(add, f64, d)
    }

    __device__ __forceinline__
    static void min(double& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(min, f64, d)
    }

    __device__ __forceinline__
    static void max(double& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(max, f64, d)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, long int> {

    __device__ __forceinline__
    static void add(long int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(add, s64, l)
    }

    __device__ __forceinline__
    static void min(long int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(min, s64, l)
    }

    __device__ __forceinline__
    static void max(long int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(max, s64, l)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, long long int> {

    __device__ __forceinline__
    static void add(long long int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(add, s64, l)
    }

    __device__ __forceinline__
    static void min(long long int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(min, s64, l)
    }

    __device__ __forceinline__
    static void max(long long int& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(max, s64, l)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, long unsigned> {

    __device__ __forceinline__
    static void add(long unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(add, u64, l)
    }

    __device__ __forceinline__
    static void min(long unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(min, u64, l)
    }

    __device__ __forceinline__
    static void max(long unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(max, u64, l)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, long long unsigned> {

    __device__ __forceinline__
    static void add(long long unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(add, u64, l)
    }

    __device__ __forceinline__
    static void min(long long unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(min, u64, l)
    }

    __device__ __forceinline__
    static void max(long long unsigned& value, unsigned mask) {
        WARP_SEG_REDUCE_MACRO2(max, u64, l)
    }
};

#undef WARP_SEG_REDUCE_MACRO
#undef WARP_SEG_REDUCE_MACRO2

} // namespace detail

//==============================================================================
//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::add(T& value, unsigned mask) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::add(value, mask);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::min(T& value, unsigned mask) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::min(value, mask);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::max(T& value, unsigned mask) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::max(value, mask);
}

//==============================================================================

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::add(T value, R* pointer, unsigned mask) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::add(value, mask);
    if (lanemask_eq() & mask)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::min(T value, R* pointer, unsigned mask) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::min(value, mask);
    if (lanemask_eq() & mask)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::max(T value, R* pointer, unsigned mask) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::max(value, mask);
    if (lanemask_eq() & mask)
        *pointer = value;
}

//==============================================================================

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicAdd(const T& value, R* pointer, unsigned mask) {
    auto value_tmp = value;
    WarpSegmentedReduce<WARP_SZ>::add(value_tmp, mask);
    if (lanemask_eq() & mask) {
        if (lane_id() != 0 && lanemask_gt() & mask)
            *pointer = value_tmp;
        else
            xlib::atomic::add(value_tmp, pointer);
    }
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicMin(const T& value, R* pointer, unsigned mask) {
    auto value_tmp = value;
    WarpSegmentedReduce::min(value_tmp, mask);
    if (lanemask_eq() & mask) {
        if (lanemask_gt() & mask) //there is no marked lanes after me
            xlib::atomic::min(value_tmp, pointer);
        else
            *pointer = value_tmp;
    }
    else if (lane_id() == 0)
        xlib::atomic::add(value_tmp, pointer);
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicMax(const T& value, R* pointer, unsigned mask) {
    auto value_tmp = value;
    WarpSegmentedReduce::max(value_tmp, mask);
    if (lanemask_eq() & mask) {
        if (lanemask_gt() & mask) //there is no marked lanes after me
            xlib::atomic::max(value_tmp, pointer);
        else
            *pointer = value_tmp;
    }
    else if (lane_id() == 0)
        xlib::atomic::add(value_tmp, pointer);
}

} // namespace xlib
