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
 *
 * @file
 */
#pragma once

#include "Host/Metaprogramming.hpp"

namespace xlib {

const unsigned     SM_THREADS = 2048;
const unsigned MAX_BLOCK_SIZE = 1024;
const unsigned MAX_BLOCK_SMEM = 49152;
const unsigned   CONSTANT_MEM = 65536;
const unsigned   MEMORY_BANKS = 32;
const unsigned      WARP_SIZE = 32;

#if (defined(__CUDACC__) && !defined(ARCH))
    #pragma error("ARCH MACRO NOT DEFINED IN CUDA PROJECT")
#endif

#if defined(ARCH)
#if ARCH >= 300
    #if ARCH <= 350
        const unsigned SMEM_PER_SM = 49152;

    #elif ARCH == 370
        const unsigned SMEM_PER_SM = 114688;

    #elif ARCH == 500 || ARCH == 530 || ARCH == 600 ||  ARCH == 620
        const unsigned SMEM_PER_SM = 65536;

    #elif ARCH == 520 || ARCH == 610
        const unsigned SMEM_PER_SM = 98304;
    #else
        #error("Unsupported Compute Cabalitity")
    #endif

    #if ARCH <= 370
        const unsigned RESIDENT_BLOCKS_PER_SM = 16;
    #else
        const unsigned RESIDENT_BLOCKS_PER_SM = 32;
    #endif

    template<typename T = char, unsigned BLOCK_SIZE = 0>
    class SMemPerThread {
        static_assert(BLOCK_SIZE == 0 || xlib::is_power2(BLOCK_SIZE),
                      "BLOCK_SIZE must be a power of 2");
        static const unsigned _BLOCK_SIZE = BLOCK_SIZE == 0 ? MAX_BLOCK_SIZE :
                                            BLOCK_SIZE;
        static const unsigned SMEM_PER_THREAD = SMEM_PER_SM / SM_THREADS;
        //max block size for full occupancy
        static const unsigned  SM_BLOCKS = SM_THREADS / _BLOCK_SIZE;
        static const unsigned OCC_RATIO1 = SM_BLOCKS / RESIDENT_BLOCKS_PER_SM;
        static const unsigned  OCC_RATIO = xlib::max(OCC_RATIO1, 1u);
        static const unsigned   SMEM_OCC = SMEM_PER_THREAD * OCC_RATIO;

        static const unsigned SMEM_LIMIT = MAX_BLOCK_SMEM / _BLOCK_SIZE;
    public:
        static const unsigned value = xlib::min(SMEM_LIMIT, SMEM_OCC) /
                                      sizeof(T);
    };

    template<typename T = char, unsigned BLOCK_SIZE = 0>
    struct SMemPerWarp {
        static const unsigned value = SMemPerThread<T, BLOCK_SIZE>::value *
                                      WARP_SIZE;
    };

    template<unsigned BLOCK_SIZE, typename T = char>
    struct SMemPerBlock {
        static const unsigned value = SMemPerThread<T, BLOCK_SIZE>::value *
                                      BLOCK_SIZE;
    };
#else
    #error("Unsupported Compute Cabalitity (CC < 3.0)")
#endif
#else
    #error("ARCH macro not defined. Please set the ARCH macro e.g. -DARCH 350")
#endif
#undef ARCH

#if defined(SM)
    const unsigned           NUM_SM = SM;
    const unsigned RESIDENT_THREADS = SM * SM_THREADS;
    const unsigned   RESIDENT_WARPS = RESIDENT_THREADS / 32;

    template<unsigned BLOCK_SIZE>
    struct ResidentBlocks {
        static const unsigned value = RESIDENT_THREADS / BLOCK_SIZE;
    };
#endif
#undef SM

} // namespace xlib
