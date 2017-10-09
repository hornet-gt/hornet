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
 */
#include "Device/Atomic.cuh"        //xlib::atomicAdd
#include "Device/Basic.cuh"         //xlib::member_mask, xlib::shfl_xor
#include "Device/PTX.cuh"           //xlib::lane_id
#include "Host/Metaprogramming.hpp" //xlib::Log2

namespace xlib {
namespace detail {

#define WARP_REDUCE_MACRO(ASM_OP, ASM_T, ASM_CL)                               \
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();                 \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < Log2<VW_SIZE>::value; STEP++) {                  \
        const int    MASK_WARP = (1 << (STEP + 1)) - 1;                        \
        const int MIN_MAX_LANE = ((31 - MASK_WARP) << 8) | MASK_WARP;          \
        asm(                                                                   \
            "{"                                                                \
            ".reg ."#ASM_T" r1;"                                               \
            ".reg .pred p;"                                                    \
            "shfl.sync.down.b32 r1|p, %1, %2, %3, %4;"                         \
            "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                \
            "mov.b32 %0, r1;"                                                  \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(1 << STEP),              \
              "r"(MIN_MAX_LANE), "r"(member_mask));                            \
    }

#define WARP_REDUCE_MACRO2(ASM_OP, ASM_T, ASM_CL)                              \
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();                 \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < Log2<VW_SIZE>::value; STEP++) {                  \
        const int    MASK_WARP = (1 << (STEP + 1)) - 1;                        \
        const int MIN_MAX_LANE = ((31 - MASK_WARP) << 8) | MASK_WARP;          \
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
              "r"(MIN_MAX_LANE), "r"(member_mask));                            \
}

//==============================================================================

template<unsigned VW_SIZE, typename T>
struct WarpReduceHelper {

    __device__ __forceinline__
    static void add(T& value) {
        const unsigned member_mask = xlib::member_mask<VW_SIZE>();
        #pragma unroll
        for (int STEP = 1; STEP <= VW_SIZE / 2; STEP = STEP * 2)
            value += xlib::shfl_xor(member_mask, value, STEP);
    }

    __device__ __forceinline__
    static void min(T& value) {
        const unsigned member_mask = xlib::member_mask<VW_SIZE>();
        #pragma unroll
        for (int STEP = 1; STEP <= VW_SIZE / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_xor(member_mask, value, STEP);
            value    = tmp < value ? tmp : value;
        }
    }

    __device__ __forceinline__
    static void max(T& value) {
        const unsigned member_mask = xlib::member_mask<VW_SIZE>();
        #pragma unroll
        for (int STEP = 1; STEP <= VW_SIZE / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_xor(member_mask, value, STEP);
            value    = tmp > value ? tmp : value;
        }
    }

    template<typename Lambda>
     __device__ __forceinline__
    static void apply(T& value, const Lambda& lambda) {
        const unsigned member_mask = xlib::member_mask<VW_SIZE>();
        #pragma unroll
        for (int STEP = 1; STEP <= VW_SIZE / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_xor(member_mask, value, STEP);
            value    = lambda(value, tmp);
        }
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, int> {

    __device__ __forceinline__
    static void add(int& value) {
        WARP_REDUCE_MACRO(add, s32, r)
    }

    __device__ __forceinline__
    static void min(int& value) {
        WARP_REDUCE_MACRO(min, s32, r)
    }

    __device__ __forceinline__
    static void max(int& value) {
        WARP_REDUCE_MACRO(max, s32, r)
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, unsigned> {

    __device__ __forceinline__
    static void add(unsigned& value) {
        WARP_REDUCE_MACRO(add, u32, r)
    }

    __device__ __forceinline__
    static void min(unsigned& value) {
        WARP_REDUCE_MACRO(min, u32, r)
    }

    __device__ __forceinline__
    static void max(unsigned& value) {
        WARP_REDUCE_MACRO(max, u32, r)
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, float> {

    __device__ __forceinline__
    static void add(float& value) {
        WARP_REDUCE_MACRO(add, f32, f)
    }

    __device__ __forceinline__
    static void min(float& value) {
        WARP_REDUCE_MACRO(min, f32, f)
    }

    __device__ __forceinline__
    static void max(float& value) {
        WARP_REDUCE_MACRO(max, f32, f)
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, double> {

    __device__ __forceinline__
    static  void add(double& value) {
        WARP_REDUCE_MACRO2(add, f64, d)
    }

    __device__ __forceinline__
    static void min(double& value) {
        WARP_REDUCE_MACRO2(min, f64, d)
    }

    __device__ __forceinline__
    static void max(double& value) {
        WARP_REDUCE_MACRO2(max, f64, d)
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, long int> {

    __device__ __forceinline__
    static  void add(long int& value) {
        WARP_REDUCE_MACRO2(add, s64, l)
    }

    __device__ __forceinline__
    static void min(long int& value) {
        WARP_REDUCE_MACRO2(min, s64, l)
    }

    __device__ __forceinline__
    static void max(long int& value) {
        WARP_REDUCE_MACRO2(max, s64, l)
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, long long int> {

    __device__ __forceinline__
    static  void add(long long int& value) {
        WARP_REDUCE_MACRO2(add, s64, l)
    }

    __device__ __forceinline__
    static void min(long long int& value) {
        WARP_REDUCE_MACRO2(min, s64, l)
    }

    __device__ __forceinline__
    static void max(long long int& value) {
        WARP_REDUCE_MACRO2(max, s64, l)
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, long unsigned> {

    __device__ __forceinline__
    static  void add(long unsigned& value) {
        WARP_REDUCE_MACRO2(add, u64, l)
    }

    __device__ __forceinline__
    static void min(long unsigned& value) {
        WARP_REDUCE_MACRO2(min, u64, l)
    }

    __device__ __forceinline__
    static void max(long unsigned& value) {
        WARP_REDUCE_MACRO2(max, u64, l)
    }
};

//------------------------------------------------------------------------------

template<unsigned VW_SIZE>
struct WarpReduceHelper<VW_SIZE, long long unsigned> {

    __device__ __forceinline__
    static  void add(long long unsigned& value) {
        WARP_REDUCE_MACRO2(add, u64, l)
    }

    __device__ __forceinline__
    static void min(long long unsigned& value) {
        WARP_REDUCE_MACRO2(min, u64, l)
    }

    __device__ __forceinline__
    static void max(long long unsigned& value) {
        WARP_REDUCE_MACRO2(max, u64, l)
    }
};

#undef WARP_REDUCE_MACRO
#undef WARP_REDUCE_MACRO2

} // namespace detail

//==============================================================================
//==============================================================================

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::add(T& value) {
    detail::WarpReduceHelper<VW_SIZE, T>::add(value);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::min(T& value) {
    detail::WarpReduceHelper<VW_SIZE, T>::min(value);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::max(T& value) {
    detail::WarpReduceHelper<VW_SIZE, T>::max(value);
}

//==============================================================================

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::addAll(T& value) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    detail::WarpReduceHelper<VW_SIZE, T>::add(value);
    value = xlib::shfl(member_mask, value, 0, VW_SIZE);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::minAll(T& value) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    detail::WarpReduceHelper<VW_SIZE, T>::min(value);
    value = xlib::shfl(member_mask, value, 0, VW_SIZE);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::maxAll(T& value) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    detail::WarpReduceHelper<VW_SIZE, T>::max(value);
    value = xlib::shfl(member_mask, value, 0, VW_SIZE);
}

//==============================================================================

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::add(T& value, R* pointer) {
    detail::WarpReduceHelper<VW_SIZE, T>::add(value);
    if (lane_id<VW_SIZE>() == 0)
        *pointer = value;
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::min(T& value, R* pointer) {
    detail::WarpReduceHelper<VW_SIZE, T>::min(value);
    if (lane_id<VW_SIZE>() == 0)
        *pointer = value;
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::max(T& value, R* pointer) {
    detail::WarpReduceHelper<VW_SIZE, T>::max(value);
    if (lane_id<VW_SIZE>() == 0)
        *pointer = value;
}

//==============================================================================

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
T WarpReduce<VW_SIZE>::atomicAdd(const T& value, R* pointer) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    T old, value_tmp = value;
    detail::WarpReduceHelper<VW_SIZE, T>::add(value_tmp);
    if (lane_id<VW_SIZE>() == 0)
        old = atomic::add(value_tmp, pointer);
    return xlib::shfl(member_mask, old, 0, VW_SIZE);
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::atomicMin(const T& value, R* pointer) {
    T value_tmp = value;
    detail::WarpReduceHelper<VW_SIZE, T>::min(value_tmp);
    if (lane_id<VW_SIZE>() == 0)
        atomic::min(value_tmp, pointer);
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::atomicMax(const T& value, R* pointer) {
    T value_tmp = value;
    detail::WarpReduceHelper<VW_SIZE, T>::max(value_tmp);
    if (lane_id<VW_SIZE>() == 0)
        atomic::max(value_tmp, pointer);
}

} // namespace xlib
