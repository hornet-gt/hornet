/**
 * @internal
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
#include "Device/Definition.cuh"

namespace xlib {
namespace detail {

#define warpInclusiveScanMACRO(ASM_OP, ASM_T, ASM_CL)                          \
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;                 \
    const unsigned member_mask = WARP_SZ == WARP_SIZE ? 0xFFFFFFFF :           \
              (0xFFFFFFFF >> (32 - WARP_SZ)) << (xlib::lane_id() / WARP_SZ);   \
    _Pragma("unroll")                                                          \
    for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {                 \
        asm(                                                                   \
            "{"                                                                \
            ".reg ."#ASM_T" r1;"                                               \
            ".reg .pred p;"                                                    \
            "shfl.sync.up.b32 r1|p, %1, %2, %3, %4;"                           \
            "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                \
            "mov."#ASM_T" %0, r1;"                                             \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(STEP), "r"(MASK),        \
              "r"(member_mask));                                               \
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
    if (xlib::lane_id() == WARP_SZ - 1)
        *pointer = value;
}

//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T& value) {
    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    const unsigned member_mask = WARP_SZ == WARP_SIZE ? 0xFFFFFFFF :
                (0xFFFFFFFF >> (32 - WARP_SZ)) << (xlib::lane_id() / WARP_SZ);
    asm(
        "{"
        ".reg .pred p;"
        "shfl.sync.up.b32 %0|p, %1, %2, %3, %4;"
        "@!p mov.b32 %0, 0;"
        "}"
        : "=r"(value) : "r"(value), "r"(1), "r"(MASK), "r"(member_mask));
}


template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T& value, T& total) {
    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    const unsigned member_mask = WARP_SZ == WARP_SIZE ? 0xFFFFFFFF :
                (0xFFFFFFFF >> (32 - WARP_SZ)) << (xlib::lane_id() / WARP_SZ);
    total = value;
    asm(
        "{"
        ".reg .pred p;"
        "shfl.sync.up.b32 %0|p, %1, %2, %3, %4;"
        "@!p mov.b32 %0, 0;"
        "}"
        : "=r"(value) : "r"(value), "r"(1), "r"(MASK), "r"(member_mask));
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::AddBcast(T& value, T& total) {
    const unsigned member_mask = WARP_SZ == WARP_SIZE ? 0xFFFFFFFF :
                (0xFFFFFFFF >> (32 - WARP_SZ)) << (xlib::lane_id() / WARP_SZ);

    detail::WarpInclusiveScanHelper<WARP_SZ, false, T>::Add(value);
    total = __shfl_sync(member_mask, value, WARP_SZ - 1, WARP_SZ);
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    asm(
        "{"
        ".reg .pred p;"
        "shfl.sync.up.b32 %0|p, %1, %2, %3, %4;"
        "@!p mov.b32 %0, 0;"
        "}"
        : "=r"(value) : "r"(value), "r"(1), "r"(MASK), "r"(member_mask));
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T& value, T* total_ptr) {
    T total;
    WarpExclusiveScan<WARP_SZ>::Add(value, total);
    if (xlib::lane_id() == WARP_SZ - 1)
        *total_ptr = total;
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
T WarpExclusiveScan<WARP_SZ>::AtomicAdd(T& value, T* total_ptr) {
    const unsigned member_mask = WARP_SZ == WARP_SIZE ? 0xFFFFFFFF :
                (0xFFFFFFFF >> (32 - WARP_SZ)) << (xlib::lane_id() / WARP_SZ);
    T total, old;
    WarpExclusiveScan<WARP_SZ>::Add(value, total);
    if (xlib::lane_id() == WARP_SZ - 1)
        old = atomicAdd(total_ptr, total);
    return __shfl_sync(member_mask, old, WARP_SZ - 1);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
T WarpExclusiveScan<WARP_SZ>::AtomicAdd(T& value, T* total_ptr, T& total) {
    const unsigned member_mask = WARP_SZ == WARP_SIZE ? 0xFFFFFFFF :
                (0xFFFFFFFF >> (32 - WARP_SZ)) << (xlib::lane_id() / WARP_SZ);
    T old;
    WarpExclusiveScan<WARP_SZ>::AddBcast(value, total);
    if (xlib::lane_id() == WARP_SZ - 1)
        old = atomicAdd(total_ptr, total);
    return __shfl_sync(member_mask, old, WARP_SZ - 1);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::Add(T* in_ptr, T* total_ptr) {
    T value = in_ptr[xlib::lane_id()];
    WarpExclusiveScan<WARP_SZ>::Add(value, total_ptr);
    in_ptr[xlib::lane_id()] = value;
}

} // namespace xlib
