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

#include <type_traits>

/** \namespace PTX
 *  provide simple interfaces for low-level PTX instructions
 */
namespace xlib {
/*
#if defined(_WIN32) || defined(__i386__)
    //#define ASM_PTR "r"
    #pragma error(ERR_START "32-bit architectures are not supported" ERR_END)
#elif defined(__x86_64__) || defined(__ia64__) ||                              \
      defined(_WIN64) || defined(__ppc64__)
    #define ASM_PTR "l"
#endif*/

// ---------------------------- THREAD PTX -------------------------------------

/**
 *  @brief return the lane ID within the current warp
 *
 *  Provide the thread ID within the current warp (called lane).
 *  \return identification ID in the range 0 &le; ID &le; 31
 */
__device__ __forceinline__ unsigned lane_id();

/**
 *  @brief terminate the current thread
 */
__device__ __forceinline__ void thread_exit();

// --------------------------------- MATH --------------------------------------

/**
 *  @brief sum three operands with one instruction
 *
 *  Sum three operand with one instruction. Only in Maxwell architecture
 *  IADD3 is implemented in hardware, otherwise involves multiple instructions.
 *  \return x + y + z
 */
__device__ __forceinline__
unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z);

/** @fn unsigned int __msb(unsigned int word)
 *  @brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 31.
 *  0xFFFFFFFF if no bit is found.
 */

/** @fn unsigned int __msb(unsigned long long int dword)
 *  @brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 63.
 *          0xFFFFFFFF if no bit is found.
 */
//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__msb(T word);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, unsigned>::type
__msb(T dword);

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) < 8, unsigned>::type
__be(T word, unsigned pos);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__be(T dword, unsigned pos);

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) < 8, T>::type
__bi(T word, unsigned pos);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, T>::type
__bi(T dword, unsigned pos);

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__bfe(T word, unsigned pos, unsigned length);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__bfe(T dword, unsigned pos, unsigned length);

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8>::type
__bfi(T& word, unsigned bitmask, unsigned pos, unsigned length);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8>::type
__bfi(T& dword, long long unsigned bitmask, unsigned pos, unsigned length);

//------------------------------------------------------------------------------

template<typename T, int _DATA_SIZE>
struct WordArray {
    const unsigned WORD_SIZE = sizeof(T) * 8;
    const unsigned DATA_SIZE = _DATA_SIZE;
public:
    WordArray(T* array) : _array(array) {}

    __device__ __forceinline__ T operator[](int index) const {
        unsigned     start = index * DATA_SIZE;
        unsigned       end = start + DATA_SIZE;
        unsigned start_pos = start / WORD_SIZE;
        unsigned mod_start = start % WORD_SIZE;
        auto         data1 = __bfe(_array[start_pos], mod_start, DATA_SIZE);
        if (start != end) {
            unsigned   head = WORD_SIZE - mod_start;
            unsigned remain = DATA_SIZE - head;
            auto      data2 = __bfe(_array[start + 1], 0, remain);
            return (data2 << head) | data1;
        }
        return data1;
    }
    __device__ __forceinline__ void insert(T data, int index) {
        unsigned     start = index * DATA_SIZE;
        unsigned       end = start + DATA_SIZE;
        unsigned start_pos = start / WORD_SIZE;
        unsigned mod_start = start % WORD_SIZE;
        auto         data1 = __bfi(_array[start_pos], data, mod_start,
                                   DATA_SIZE);
        if (start != end) {
            unsigned   head = WORD_SIZE - mod_start;
            unsigned remain = DATA_SIZE - head;
            auto      data2 = __bfi(_array[start + 1], data >> head, 0, remain);
        }
    }
private:
    T* _array;
};

/** @fn unsigned int LaneMaskEQ()
 *  @brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return 1 << laneid
 */
__device__ __forceinline__ unsigned LaneMaskEQ();

/** @fn unsigned int LaneMaskLT()
 *  @brief 32-bit mask with bits set in positions less than the thread's lane
 *         number in the warp
 *  \return (1 << laneid) - 1
 */
__device__ __forceinline__ unsigned LaneMaskLT();

/** @fn unsigned int LaneMaskLE()
 *  @brief 32-bit mask with bits set in positions less than or equal to the
 *         thread's lane number in the warp
 *  \return (1 << (laneid + 1)) - 1
 */
__device__ __forceinline__ unsigned LaneMaskLE();

/** @fn unsigned int LaneMaskGT()
 *  @brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return ~((1 << (laneid + 1)) - 1)
 */
__device__ __forceinline__ unsigned LaneMaskGT();

/** @fn unsigned int LaneMaskGE()
 *  @brief 32-bit mask with bits set in positions greater than or equal to the
 *         thread's lane number in the warp
 *  \return ~((1 << laneid) - 1)
 */
__device__ __forceinline__ unsigned LaneMaskGE();

} // namespace xlib

#include "impl/PTX.i.cuh"
