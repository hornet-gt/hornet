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
namespace xlib {
namespace detail {

#define warpInclusiveScanMACRO(ASM_OP, ASM_T, ASM_CL)                          \
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;                 \
    _Pragma("unroll")                                                          \
    for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {                 \
        asm(                                                                   \
            "{"                                                                \
            ".reg ."#ASM_T" r1;"                                               \
            ".reg .pred p;"                                                    \
            "shfl.up.b32 r1|p, %1, %2, %3;"                                    \
            "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                \
            "mov."#ASM_T" %0, r1;"                                             \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(STEP), "r"(MASK));       \
    }

//==============================================================================

template<int WARP_SZ, bool BROADCAST, typename T>
struct WarpInclusiveScanHelper {
    static __device__ __forceinline__ void Add(T& value);
};

template<int WARP_SZ, bool BROADCAST>
struct WarpInclusiveScanHelper<WARP_SZ, BROADCAST, int> {
    static __device__ __forceinline__ void Add(int& value) {
        warpInclusiveScanMACRO(add, s32, r)
    }
};

template<int WARP_SZ, bool BROADCAST>
struct WarpInclusiveScanHelper<WARP_SZ, BROADCAST, float> {
    static __device__ __forceinline__ void Add(float& value) {
        warpInclusiveScanMACRO(add, f32, f)
    }
};
#undef warpInclusiveScanMACRO
} // namespace detail

//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpInclusiveScan<WARP_SZ>::Add(T& value) {
    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpInclusiveScan<WARP_SZ>::Add(T& value, T& total) {
    detail::WarpInclusiveScanHelper<WARP_SZ, true, T>::Add(value);
    total = __shfl(value, WARP_SZ - 1, WARP_SZ);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpInclusiveScan<WARP_SZ>::Add(T& value, T* pointer) {

    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
    if (lane_id() == WARP_SZ - 1)
        *pointer = value;
}

//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T& value) {
    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    asm(
        "{"
        ".reg .pred p;"
        "shfl.up.b32 %0|p, %1, %2, %3;"
        "@!p mov.b32 %0, 0;"
        "}"
        : "=r"(value) : "r"(value), "r"(1), "r"(MASK));
}


template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T& value, T& total) {
    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    total = value;
    asm(
        "{"
        ".reg .pred p;"
        "shfl.up.b32 %0|p, %1, %2, %3;"
        "@!p mov.b32 %0, 0;"
        "}"
        : "=r"(value) : "r"(value), "r"(1), "r"(MASK));
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::AddBcast(T& value, T& total) {
    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
    total = __shfl(value, WARP_SZ - 1, WARP_SZ);
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    asm(
        "{"
        ".reg .pred p;"
        "shfl.up.b32 %0|p, %1, %2, %3;"
        "@!p mov.b32 %0, 0;"
        "}"
        : "=r"(value) : "r"(value), "r"(1), "r"(MASK));
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T& value, T* total_ptr) {
    T total;
    WarpExclusiveScan<WARP_SZ>::Add(value, total);
    if (lane_id() == WARP_SZ - 1)
        *total_ptr = total;
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
T WarpExclusiveScan<WARP_SZ>::AtomicAdd(T& value, T* total_ptr) {
    T total, old;
    WarpExclusiveScan<WARP_SZ>::Add(value, total);
    if (lane_id() == WARP_SZ - 1)
        old = atomicAdd(total_ptr, total);
    return __shfl(old, WARP_SZ - 1);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
T WarpExclusiveScan<WARP_SZ>::AtomicAdd(T& value, T* total_ptr, T& total) {
    T old;
    WarpExclusiveScan<WARP_SZ>::AddBcast(value, total);
    if (lane_id() == WARP_SZ - 1)
        old = atomicAdd(total_ptr, total);
    return __shfl(old, WARP_SZ - 1);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T* in_ptr, T* total_ptr) {
    T value = in_ptr[lane_id()];
    WarpExclusiveScan<WARP_SZ>::Add(value, total_ptr);
    in_ptr[lane_id()] = value;
}

} // namespace xlib
