/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include "Base/Device/Util/PTX.cuh"
#include "Base/Device/Util/Atomic.cuh"
#include "Base/Host/Basic.hpp"

namespace xlib {
namespace detail {

#define warpReduceMACRO(ASM_OP, ASM_T, ASM_CL)                                 \
_Pragma("unroll")                                                              \
for (int STEP = 0; STEP < Log2<WARP_SZ>::value; STEP++) {                      \
    const int MASK_WARP = (1 << (STEP + 1)) - 1;                               \
    const int C = ((31 - MASK_WARP) << 8) | (MASK_WARP) ;                      \
    asm(                                                                       \
        "{"                                                                    \
        ".reg ."#ASM_T" r1;"                                                   \
        ".reg .pred p;"                                                        \
        "shfl.down.b32 r1|p, %1, %2, %3;"                                      \
        "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                    \
        "mov."#ASM_T" %0, r1;"                                                 \
        "}"                                                                    \
        : "="#ASM_CL(value) : #ASM_CL(value), "r"(1 << STEP), "r"(C));         \
}                                                                              \
if (BROADCAST)                                                                 \
    value = __shfl(value, 0, WARP_SZ);

#define warpReduceMACRO2(ASM_OP1, ASM_T1, ASM_CL1, ASM_OP2, ASM_T2, ASM_CL2)   \
_Pragma("unroll")                                                              \
for (int STEP = 0; STEP < Log2<WARP_SZ>::value; STEP++) {                      \
    const int MASK_WARP = (1 << (STEP + 1)) - 1;                               \
    const int C = ((31 - MASK_WARP) << 8) | (MASK_WARP) ;                      \
    asm(                                                                       \
        "{"                                                                    \
        ".reg ."#ASM_T1" r1;"                                                  \
        ".reg ."#ASM_T2" r2;"                                                  \
        ".reg .pred p;"                                                        \
        ".reg .pred q;"                                                        \
        "shfl.down.b32 r1|p, %2, %4, %5;"                                      \
        "shfl.down.b32 r2|q, %3, %4, %5;"                                      \
        "@p "#ASM_OP1"."#ASM_T1" r1, r1, %2;"                                  \
        "@q "#ASM_OP2"."#ASM_T2" r2, r2, %3;"                                  \
        "mov."#ASM_T1" %0, r1;"                                                \
        "mov."#ASM_T2" %1, r2;"                                                \
        "}"                                                                    \
        : "="#ASM_CL1(value1), "="#ASM_CL2(value2) :                           \
             #ASM_CL1(value1), #ASM_CL2(value2), "r"(1 << STEP), "r"(C));      \
}                                                                              \
if (BROADCAST) {                                                               \
    value1 = __shfl(value1, 0, WARP_SZ);                                       \
    value2 = __shfl(value2, 0, WARP_SZ);                                       \
}

//==============================================================================

template<int WARP_SZ, bool BROADCAST, typename T>
struct WarpReduceHelper {
    static __device__ __forceinline__ void add(T& value) {
        #pragma unroll
        for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
            value += xlib::shfl_xor(value, i);
    }

    static __device__ __forceinline__ void min(T& value) {
        #pragma unroll
        for (int i = WARP_SIZE / 2; i >= 1; i /= 2) {
            auto tmp = xlib::shfl_xor(value, i);
            value    = tmp < value ? tmp : value;
        }
    }

    static __device__ __forceinline__ void max(T& value) {
        #pragma unroll
        for (int i = WARP_SIZE / 2; i >= 1; i /= 2) {
            auto tmp = xlib::shfl_xor(value, i);
            value    = tmp > value ? tmp : value;
        }
    }

    /*template<typename Lambda>
    static __device__ __forceinline__ void apply(T& value,
                                                 const Lambda& lambda) {
        #pragma unroll
        for (int i = WARP_SIZE / 2; i >= 1; i /= 2) {
            auto tmp = xlib::shfl_xor(value, i);
            lambda(value, tmp);
        }
    }*/
};

template<int WARP_SZ, bool BROADCAST>
struct WarpReduceHelper<WARP_SZ, BROADCAST, int> {
    static __device__ __forceinline__ void add(int& value) {
        warpReduceMACRO(add, s32, r)
    }

    static __device__ __forceinline__ void min(int& value) {
        warpReduceMACRO(min, s32, r)
    }

    static __device__ __forceinline__ void max(int& value) {
        warpReduceMACRO(max, s32, r)
    }

    /*static __device__ __forceinline__ void add(int& value1, int& value2) {
        warpReduceMACRO2(add, s32, r, add, s32, r)
    }*/
};

template<int WARP_SZ, bool BROADCAST>
struct WarpReduceHelper<WARP_SZ, BROADCAST, float> {
    static __device__ __forceinline__ void add(float& value) {
        warpReduceMACRO(add, f32, f)
    }

    static __device__ __forceinline__ void min(float& value) {
        warpReduceMACRO(min, f32, f)
    }

    static __device__ __forceinline__ void max(float& value) {
        warpReduceMACRO(max, f32, f)
    }
};

template<int WARP_SZ, bool BROADCAST>
struct WarpReduceHelper<WARP_SZ, BROADCAST, unsigned> {
    static __device__ __forceinline__ void add(unsigned& value) {
        warpReduceMACRO(add, u32, r)
    }

    static __device__ __forceinline__ void min(unsigned& value) {
        warpReduceMACRO(min, u32, r)
    }

    static __device__ __forceinline__ void max(unsigned& value) {
        warpReduceMACRO(max, u32, r)
    }
};

#undef warpReduceMACRO
#undef warpReduceMACRO2
} // namespace detail

//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::add(T& value) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::add(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::min(T& value) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::min(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::max(T& value) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::max(value);
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::addAll(T& value) {
    detail::WarpReduceHelper<WARP_SZ, true, T>::add(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::minAll(T& value) {
    detail::WarpReduceHelper<WARP_SZ, true, T>::min(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::maxAll(T& value) {
    detail::WarpReduceHelper<WARP_SZ, true, T>::max(value);
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::add(T& value, T* pointer) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::add(value);
    if (lane_id() == 0)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::min(T& value, T* pointer) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::min(value);
    if (lane_id() == 0)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::max(T& value, T* pointer) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::max(value);
    if (lane_id() == 0)
        *pointer = value;
}

//------------------------------------------------------------------------------
/*
template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::AtomicAdd(T& value, T* pointer) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::Add(value);
    if (lane_id() == 0)
        atomicAdd(pointer, value);
}*/

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
T WarpReduce<WARP_SZ>::atomicAdd(const T& value, T* pointer) {
    T old;
    auto value_tmp = value;
    detail::WarpReduceHelper<WARP_SZ, false, T>::add(value_tmp);
    if (lane_id() == 0)
        old = xlib::atomic::max(value_tmp, pointer); //::atomicAdd(pointer, value_tmp);
    return __shfl(old, 0, WARP_SZ);
}

/*
template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
T WarpReduce<WARP_SZ>::rAtomicAdd(T& value, T* pointer) {
    T old;
    detail::WarpReduceHelper<WARP_SZ, false, T>::Add(value);
    if (lane_id() == 0)
        old = atomicAdd(pointer, value);
    return __shfl(old, 0, WARP_SZ);
}*/

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::atomicMin(const T& value, T* pointer) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
     "Atomic operations on types of size different from 4/8 are not supported");

    auto value_tmp = value;
    detail::WarpReduceHelper<WARP_SZ, false, T>::min(value_tmp);
    if (lane_id() == 0)
        xlib::atomic::min(value_tmp, pointer);
}
/*
template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::AtomicMax(T& value, T* pointer) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
     "Atomic operations on types of size different from 4/8 are not supported");
    using T_nocv = typename std::remove_cv<T>::type;

    detail::WarpReduceHelper<WARP_SZ, false, T>::Max(value);
    if (lane_id() == 0) {
        if (std::is_same<T_nocv, int>::value) {
            atomicMax(reinterpret_cast<int*>(pointer),
                      reinterpret_cast<int&>(value));
        }
        else if (sizeof (T) == 4) {
            atomicMax(reinterpret_cast<unsigned int*>(pointer),
                      reinterpret_cast<unsigned int&>(value));
        }
        else if (sizeof(T) == 8) {
            atomicMax(reinterpret_cast<long long unsigned int*>(pointer),
                      reinterpret_cast<long long unsigned int&>(value));
        }
    }
}*/

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::atomicMax(const T& value, T* pointer) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
     "Atomic operations on types of size different from 4/8 are not supported");

    auto value_tmp = value;
    detail::WarpReduceHelper<WARP_SZ, false, T>::max(value_tmp);
    if (lane_id() == 0)
        xlib::atomic::max(value_tmp, pointer);
}

//------------------------------------------------------------------------------
/*
template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpReduce<WARP_SZ>::AtomicAdd(T& value1, T* pointer1,
                                    T& value2, T* pointer2) {
    detail::WarpReduceHelper<WARP_SZ, false, T>::Add(value1, value2);
    if (lane_id() == 0) {
        atomicAdd(pointer1, value1);
        atomicAdd(pointer2, value2);
    }
}*/

} // namespace xlib
