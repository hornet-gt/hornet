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
#pragma once

namespace xlib {

enum class SCAN_MODE { ATOMIC, REDUCE };

/**
 * The shared memory shared_mem/smem_total is not guarantee to be safe for reuse
 * !!! Before write into shared memory it requires a syncthreads
 */
template<unsigned BLOCK_SIZE = 0, SCAN_MODE mode = SCAN_MODE::ATOMIC>
struct BlockExclusiveScan {
    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value, T* smem_total);

    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T* shared_mem);

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T* shared_mem);
};

} // namespace xlib

#include "impl/BlockScan.i.cuh"
