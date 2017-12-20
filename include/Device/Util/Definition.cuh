/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
 * @version v1.4
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
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

#include "Host/Numeric.hpp"

namespace xlib {

///@brief Total number of threads for Streaming Multiprocessor (SM)
const unsigned THREADS_PER_SM = 2048;

///@brief Maximum number of threads per block
const unsigned MAX_BLOCK_SIZE = 1024;

///@brief Maximum allocable shared memory per block (bytes)
const unsigned MAX_BLOCK_SMEM = 49152;

///@brief Total GPU constant memory (bytes)
const unsigned CONSTANT_MEM   = 65536;

///@brief Number of shared memory banks
const unsigned MEMORY_BANKS   = 32;

///@brief Number of threads in a warp
const unsigned WARP_SIZE      = 32;

//==============================================================================

/**
 * @brief Compute capability-dependent device properties.
 * @details Available properties at compile time:
 * - shared memory per Streaming Multiprocessor: @p SMEM_PER_SM
 * - maximum number of resident blocks per Streaming Multiprocessor:
 *   @p RBLOCKS_PER_SM
 * Supported architecture: *300, 320, 350, 370, 500, 520, 530, 600, 610, 620,
 *                          700*
 * @tparam CUDA_ARCH identifier of the GPU architecture (3 digits)
 */
template<int CUDA_ARCH>
struct DeviceProp {
    static_assert(CUDA_ARCH != CUDA_ARCH, "Unsupported Compute Cabalitity");
};

template<>
struct DeviceProp<300> {
    static const unsigned SMEM_PER_SM    = 49152;
    static const unsigned RBLOCKS_PER_SM = 16;
};

template<>
struct DeviceProp<320> {
    static const unsigned SMEM_PER_SM    = 49152;
    static const unsigned RBLOCKS_PER_SM = 16;
};

template<>
struct DeviceProp<370> {
    static const unsigned SMEM_PER_SM    = 114688;
    static const unsigned RBLOCKS_PER_SM = 16;
};

template<>
struct DeviceProp<500> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<520> {
    static const unsigned SMEM_PER_SM    = 98304;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<530> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<600> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<610> {
    static const unsigned SMEM_PER_SM    = 98304;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<620> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<700> {
    static const unsigned SMEM_PER_SM    = 98304;
    static const unsigned RBLOCKS_PER_SM = 32;
};

//==============================================================================

/**
 * @brief Available shared memory per thread by considering the maximum
 *        occupancy for a give block size
 * @tparam T type to allocate in the shared memory
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam R any type. It must be speficied if the kernel which use this method
 *           is not compiled and BLOCK_SIZE or T are statically known
 */
template<typename T = int8_t, unsigned BLOCK_SIZE = 0, typename R = void>
__device__ __forceinline__
constexpr unsigned smem_per_thread() {
#if defined(__CUDA_ARCH__)
    unsigned SMEM_PER_SM    = DeviceProp<__CUDA_ARCH__>::SMEM_PER_SM;
    unsigned RBLOCKS_PER_SM = DeviceProp<__CUDA_ARCH__>::RBLOCKS_PER_SM;

    unsigned _BLOCK_SIZE    = (BLOCK_SIZE == 0) ? MAX_BLOCK_SIZE : BLOCK_SIZE;
    unsigned NUM_BLOCKS     = THREADS_PER_SM / _BLOCK_SIZE;
    unsigned ACTUAL_BLOCKS  = xlib::min(RBLOCKS_PER_SM, NUM_BLOCKS);
    unsigned SMEM_PER_BLOCK = xlib::min(SMEM_PER_SM / ACTUAL_BLOCKS,
                                        MAX_BLOCK_SMEM);
    return (SMEM_PER_BLOCK / BLOCK_SIZE) / sizeof(T);
#else
    static_assert(BLOCK_SIZE != BLOCK_SIZE, "not defined");
    return 0;
#endif
};

//------------------------------------------------------------------------------

/**
 * @brief Available shared memory per warp by considering the maximum
 *        occupancy for a give block size
 * @tparam T type to allocate in the shared memory
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam R any type. it must be speficied if the kernel which use this method
 *           is not compiled and BLOCK_SIZE or T are statically known
 */
template<typename T = int8_t, unsigned BLOCK_SIZE = 0, typename R = void>
__device__ __forceinline__
constexpr unsigned smem_per_warp() {
    return xlib::smem_per_thread<T, BLOCK_SIZE>() * xlib::WARP_SIZE;
};

//------------------------------------------------------------------------------

/**
 * @brief Available shared memory per block by considering the maximum
 *        occupancy for a give blocks size
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam T type to allocate in the shared memory
 * @tparam R any type. it must be speficied if the kernel which use this method
 *           is not compiled and BLOCK_SIZE or T are statically known
 */
template<unsigned BLOCK_SIZE, typename T = int8_t, typename = void>
__device__ __forceinline__
constexpr unsigned smem_per_block() {
    return xlib::smem_per_thread<T, BLOCK_SIZE>() * BLOCK_SIZE;
};

//==============================================================================
//==============================================================================

namespace host {

/**
 * @brief Available shared memory per thread by considering the maximum
 *        occupancy for a give block size or for a give number of blocks per SM
 * @tparam CUDA_ARCH cuda compute cabalitity @ref DeviceProp
 * @tparam T type to allocate in the shared memory
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam R any type. It must be speficied if the kernel which use this method
 *           is not compiled and BLOCK_SIZE or T are statically known
 * @param num_blocks_per_SM number of blocks per SM. If zero, the method uses
                            the block size
 */
template<int CUDA_ARCH, typename T = int8_t, unsigned BLOCK_SIZE = 0>
constexpr unsigned smem_per_thread(int num_blocks_per_SM = 0) {
    static_assert(BLOCK_SIZE >= 0 && BLOCK_SIZE < MAX_BLOCK_SIZE,
                  "BLOCK_SIZE range");
    unsigned SMEM_PER_SM    = DeviceProp<CUDA_ARCH>::SMEM_PER_SM;
    unsigned RBLOCKS_PER_SM = DeviceProp<CUDA_ARCH>::RBLOCKS_PER_SM;

    assert(num_blocks_per_SM >= 0 && num_blocks_per_SM < RBLOCKS_PER_SM &&
           "num_blocks_per_SM range");
    assert(BLOCK_SIZE != 0 && num_blocks_per_SM != 0);

    unsigned _BLOCK_SIZE    = (BLOCK_SIZE == 0) ? MAX_BLOCK_SIZE : BLOCK_SIZE;
    unsigned NUM_BLOCKS     = num_blocks_per_SM != 0 ? num_blocks_per_SM :
                                THREADS_PER_SM / _BLOCK_SIZE;
    unsigned ACTUAL_BLOCKS  = xlib::min(RBLOCKS_PER_SM, NUM_BLOCKS);
    unsigned SMEM_PER_BLOCK = xlib::min(SMEM_PER_SM / ACTUAL_BLOCKS,
                                        MAX_BLOCK_SMEM);
    return (SMEM_PER_BLOCK / BLOCK_SIZE) / sizeof(T);
};

/**
 * @brief Available shared memory per warp by considering the maximum
 *        occupancy for a give block size or for a give number of blocks per SM
 * @tparam CUDA_ARCH cuda compute cabalitity @ref DeviceProp
 * @tparam T type to allocate in the shared memory
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam R any type. it must be speficied if the kernel which use this method
 *           is not compiled and BLOCK_SIZE or T are statically known
 * @param num_blocks_per_SM number of blocks per SM. If zero, the method uses
                            the block size
 */
template<typename T = int8_t, unsigned BLOCK_SIZE = 0>
constexpr unsigned smem_per_warp(int num_blocks_per_SM = 0) {
    return xlib::host::smem_per_thread<T, BLOCK_SIZE>(num_blocks_per_SM) *
            xlib::WARP_SIZE;
};

//------------------------------------------------------------------------------

/**
 * @brief Available shared memory per block by considering the maximum
 *        occupancy for a give block size or for a give number of blocks per SM
 * @tparam CUDA_ARCH cuda compute cabalitity @ref DeviceProp
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam T type to allocate in the shared memory
 * @tparam R any type. it must be speficied if the kernel which use this method
 *           is not compiled and BLOCK_SIZE or T are statically known
 * @param num_blocks_per_SM number of blocks per SM. If zero, the method uses
                            the block size
 */
template<typename T = int8_t, unsigned BLOCK_SIZE = 0>
constexpr unsigned smem_per_block(int num_blocks_per_SM = 0) {
    return xlib::host::smem_per_thread<T, BLOCK_SIZE>(num_blocks_per_SM) *
            BLOCK_SIZE;
};

} // namespace host
} // namespace xlib



/*
#if defined(ARCH) && defined(__CUDA_ARCH__) && (ARCH != __CUDA_ARCH__)
    #pragma error("IF 'ARCH' and '__CUDA_ARCH__' are defined"\
                  "they must have the same value")
#endif

#if ARCH != -1 && (ARCH < 300)
    #error("Unsupported Compute Cabalitity (CC < 3.0)")
#endif*/

//==============================================================================
/*
#if ARCH == 300 || __CUDA_ARCH__ == 300 ||                                     \
    ARCH == 350 || __CUDA_ARCH__ == 350

    const unsigned SMEM_PER_SM = 49152;

#elif ARCH == 370 || __CUDA_ARCH__ == 370

    const unsigned SMEM_PER_SM = 114688;

#elif ARCH == 500 || __CUDA_ARCH__ == 500 ||                                   \
      ARCH == 530 || __CUDA_ARCH__ == 530 ||                                   \
      ARCH == 600 || __CUDA_ARCH__ == 600 ||                                   \
      ARCH == 620 || __CUDA_ARCH__ == 620

    const unsigned SMEM_PER_SM = 65536;

#elif ARCH == 520 || __CUDA_ARCH__ == 520 ||                                   \
      ARCH == 610 || __CUDA_ARCH__ == 610 ||                                   \
      ARCH == 700 || __CUDA_ARCH__ == 700

    const unsigned SMEM_PER_SM = 98304;

#endif
//------------------------------------------------------------------------------

#if ARCH <= 370 || __CUDA_ARCH__ <= 370
    const unsigned RBLOCKS_PER_SM = 16;
#else
    const unsigned RBLOCKS_PER_SM = 32;
#endif

#undef ARCH*/
    /*static_assert(BLOCK_SIZE == 0 || xlib::is_power2(BLOCK_SIZE),
                  "BLOCK_SIZE must be a power of 2");
    unsigned SMEM_PER_SM    = DeviceProp<__CUDA_ARCH__>::SMEM_PER_SM;
    unsigned RBLOCKS_PER_SM = DeviceProp<__CUDA_ARCH__>::RBLOCKS_PER_SM;

    unsigned _BLOCK_SIZE = (BLOCK_SIZE == 0) ? MAX_BLOCK_SIZE : BLOCK_SIZE;

    unsigned SMEM_PER_THREAD = SMEM_PER_SM / THREADS_PER_SM;
    //max block size for full occupancy
    unsigned SM_BLOCKS  = THREADS_PER_SM / _BLOCK_SIZE;
    unsigned OCC_RATIO1 = SM_BLOCKS / RBLOCKS_PER_SM;
    unsigned OCC_RATIO  = xlib::max(OCC_RATIO1, 1u);
    unsigned SMEM_OCC   = SMEM_PER_THREAD * OCC_RATIO;
    unsigned SMEM_LIMIT = MAX_BLOCK_SMEM / _BLOCK_SIZE;

    return xlib::min(SMEM_LIMIT, SMEM_OCC) / sizeof(T);*/

//==============================================================================
/*
#if defined(__CUDA_ARCH__)

template<typename T = char, unsigned BLOCK_SIZE = 0>
struct SMemPerThread {
    static_assert(BLOCK_SIZE == 0 || xlib::is_power2(BLOCK_SIZE),
                  "BLOCK_SIZE must be a power of 2");

    static const unsigned _BLOCK_SIZE = BLOCK_SIZE == 0 ? MAX_BLOCK_SIZE :
                                        BLOCK_SIZE;
    static const unsigned SMEM_PER_THREAD = SMEM_PER_SM / THREADS_PER_SM;
    //max block size for full occupancy
    static const unsigned SM_BLOCKS  = THREADS_PER_SM / _BLOCK_SIZE;
    static const unsigned OCC_RATIO1 = SM_BLOCKS / RBLOCKS_PER_SM;
    static const unsigned OCC_RATIO  = xlib::max(OCC_RATIO1, 1u);
    static const unsigned SMEM_OCC   = SMEM_PER_THREAD * OCC_RATIO;

    static const unsigned SMEM_LIMIT = MAX_BLOCK_SMEM / _BLOCK_SIZE;
public:
    static const unsigned value = xlib::min(SMEM_LIMIT, SMEM_OCC) /
                                  sizeof(T);
};

//------------------------------------------------------------------------------

template<typename T = char, unsigned BLOCK_SIZE = 0>
struct SMemPerWarp {
    static const unsigned value = SMemPerThread<T, BLOCK_SIZE>::value *
                                  xlib::WARP_SIZE;
};

//------------------------------------------------------------------------------

template<unsigned BLOCK_SIZE, typename T = char>
struct SMemPerBlock {
    static const unsigned value = SMemPerThread<T, BLOCK_SIZE>::value *
                                  BLOCK_SIZE;
};

//------------------------------------------------------------------------------
#endif*/
