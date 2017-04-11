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
#include "Support/Numeric.hpp"

namespace xlib {

__device__ __forceinline__ unsigned lane_id() {
    unsigned ret;
    asm ("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}

__device__ __forceinline__ void thread_exit() {
    asm ("exit;");
}

// Three-operand add    // MAXWELL
__device__ __forceinline__
unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int ret;
    asm ("vadd.u32.u32.u32.add %0, %1, %2, %3;" :
         "=r"(ret) : "r"(x), "r"(y), "r"(z));
    return ret;
}

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__msb(T word) {
    unsigned ret;
    asm ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, unsigned>::type
__msb(T dword) {
    unsigned ret;
    asm ("bfind.u64 %0, %1;" : "=r"(ret) : "l"(dword));
    return ret;
}

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) < 8, unsigned>::type
__be(T word, unsigned pos) {
    unsigned ret;
    asm ("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) :
            "r"(reinterpret_cast<unsigned&>(word)), "r"(pos), "r"(1u));
    return ret;
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__be(T dword, unsigned pos) {
    long long unsigned ret;
    asm ("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) :
          "l"(reinterpret_cast<long long unsigned&>(dword)), "r"(pos), "r"(1u));
    return ret;
}

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) < 8, T>::type
__bi(T word, unsigned pos) {
    unsigned ret;
    asm ("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(ret)
          : "r"(1u), "r"(reinterpret_cast<unsigned&>(word)), "r"(pos), "r"(1u));
    return reinterpret_cast<T&>(ret);
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, T>::type
__bi(T dword, unsigned pos) {
    long long unsigned ret;
    asm ("bfi.b64 %0, %1, %2, %3, %4;" :
            "=l"(ret) : "l"(reinterpret_cast<long long unsigned&>(dword)),
            "l"(1ull), "r"(pos), "r"(1u));
    return reinterpret_cast<T&>(ret);
}

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__bfe(T word, unsigned pos, unsigned length) {
    unsigned ret;
    asm ("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(word),
         "r"(pos), "r"(length));
    return ret;
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__bfe(T dword, unsigned pos, unsigned length) {
    long long unsigned ret;
    asm ("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(dword),
         "r"(pos), "r"(length));
    return ret;
}

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8>::type
__bfi(T& word, unsigned bitmask, unsigned pos, unsigned length) {
    asm ("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(word) : "r"(bitmask),
         "r"(word), "r"(pos), "r"(length));
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8>::type
__bfi(T& dword, long long unsigned bitmask, unsigned pos, unsigned length) {
    asm ("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(dword) : "l"(bitmask),
         "l"(dword), "r"(pos), "r"(length));
}

//------------------------------------------------------------------------------

__device__ __forceinline__ unsigned int LaneMaskEQ() {
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_eq;" : "=r"(ret) );
    return ret;
}

__device__ __forceinline__ unsigned int LaneMaskLT() {
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
    return ret;
}

__device__ __forceinline__ unsigned int LaneMaskLE() {
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_le;" : "=r"(ret) );
    return ret;
}

__device__ __forceinline__ unsigned int LaneMaskGT() {
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_gt;" : "=r"(ret) );
    return ret;
}

__device__ __forceinline__ unsigned int LaneMaskGE() {
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_ge;" : "=r"(ret) );
    return ret;
}

} // namespace xlib
