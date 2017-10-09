/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
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
#include "Device/Atomic.cuh"
#include "Device/Basic.cuh"
#include "Device/PTX.cuh"

namespace xlib {

#define WARP_SEG_REDUCE_MACRO(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<xlib::WARP_SIZE>::value; STEP++) {    \
        asm(                                                                   \
            "{"                                                                \
            ".reg ."#ASM_T" r1;"                                               \
            ".reg .pred p;"                                                    \
            "shfl.sync.down.b32 r1|p, %1, %2, %3, %4;"                         \
            "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                \
            "mov."#ASM_T" %0, r1;"                                             \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(1 << STEP),              \
              "r"(max_lane), "r"(0xFFFFFFFF));                                 \
    }

//==============================================================================
//==============================================================================

template<>
__device__ __forceinline__
void WarpSegmentedReduce::add<int>(int& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(add, s32, r)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::add<unsigned>(unsigned& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(add, u32, r)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::add<float>(float& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(add, f32, f)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::add<double>(double& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(add, f64, d)
}

//==============================================================================

template<>
__device__ __forceinline__
void WarpSegmentedReduce::min<int>(int& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(min, s32, r)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::min<unsigned>(unsigned& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(min, u32, r)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::min<float>(float& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(min, f32, f)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::min<double>(double& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(min, f64, d)
}

//==============================================================================

template<>
__device__ __forceinline__
void WarpSegmentedReduce::max<int>(int& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(max, s32, r)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::max<unsigned>(unsigned& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(max, u32, r)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::max<float>(float& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(max, f32, f)
}

template<>
__device__ __forceinline__
void WarpSegmentedReduce::max<double>(double& value, unsigned mask) {
    unsigned max_lane = __clz(__brev((xlib::lanemask_gt() & mask))) - 1;
    WARP_SEG_REDUCE_MACRO(max, f64, d)
}
#undef WARP_SEG_REDUCE_MACRO
//==============================================================================
//==============================================================================

template<typename T, typename R>
__device__ __forceinline__
T WarpSegmentedReduce::atomicAdd(const T& value, R* pointer, unsigned mask) {
    auto value_tmp = value;
    WarpSegmentedReduce::add(value_tmp, mask);
    if (lanemask_eq() & mask) {
        if (lanemask_gt() & mask) //there is no marked lanes after me
            xlib::atomic::add(value_tmp, pointer);
        else
            *pointer = value_tmp;
    }
    else if (lane_id() == 0)
        xlib::atomic::add(value_tmp, pointer);
}

template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce::atomicMin(const T& value, R* pointer, unsigned mask) {
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

template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce::atomicMax(const T& value, R* pointer,
                                    unsigned mask) {
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
