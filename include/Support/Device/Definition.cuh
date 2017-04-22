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

#include "Support/Host/Metaprogramming.hpp"

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
        static_assert(BLOCK_SIZE == 0 || IsPower2<BLOCK_SIZE>::value,
                      "BLOCK_SIZE must be a power of 2");
        static const unsigned _BLOCK_SIZE = BLOCK_SIZE == 0 ? MAX_BLOCK_SIZE :
                                            BLOCK_SIZE;
        static const unsigned SMEM_PER_THREAD = SMEM_PER_SM / SM_THREADS;
        //max block size for full occupancy
        static const unsigned  SM_BLOCKS = SM_THREADS / _BLOCK_SIZE;
        static const unsigned OCC_RATIO1 = SM_BLOCKS / RESIDENT_BLOCKS_PER_SM;
        static const unsigned  OCC_RATIO = Max<OCC_RATIO1, 1>::value;
        static const unsigned   SMEM_OCC = SMEM_PER_THREAD * OCC_RATIO;

        static const unsigned SMEM_LIMIT = MAX_BLOCK_SMEM / _BLOCK_SIZE;
    public:
        static const unsigned value = Min<SMEM_LIMIT, SMEM_OCC>::value /
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
