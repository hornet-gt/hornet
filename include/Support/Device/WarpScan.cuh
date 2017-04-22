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

#include "Support/Host/Basic.hpp"

namespace xlib {

template<int WARP_SZ = 32>
struct WarpInclusiveScan {
    /// @cond
    static_assert(IsPower2<WARP_SZ>::value &&
                  WARP_SZ >= 1 && WARP_SZ <= 32,
                  "WarpInclusiveScan : WARP_SZ must be a power of 2\
                                       and 2 <= WARP_SZ <= 32");
    /// @endcond

    template<typename T>
    static __device__ __forceinline__ void Add(T& value);

    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T& total);

    template<typename T>
    static __device__ __forceinline__ void AddBcast(T& value, T& total_ptr);

    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T* total_ptr);
};

//------------------------------------------------------------------------------

/** \struct WarpExclusiveScan WarpScan.cuh
 *  \brief Support structure for warp-level exclusive scan
 *  <pre>
 *  Input:  1 2 3 4
 *  Output: 0 1 3 6 (10)
 *  </pre>
 *  \callergraph \callgraph
 *  @pre WARP_SZ must be a power of 2 in the range 1 &le; WARP_SZ &le; 32
 *  @tparam WARP_SZ     split the warp in WARP_SIZE / WARP_SZ groups and
 *                      perform the exclusive prefix-scan in each groups.
 *                      Require log2 ( WARP_SZ ) steps
 */
template<int WARP_SZ = 32>
struct WarpExclusiveScan {
    /// @cond
    static_assert(IsPower2<WARP_SZ>::value &&
                  WARP_SZ >= 2 && WARP_SZ <= 32,
                  "WarpExclusiveScan : WARP_SZ must be a power of 2\
                             and 2 <= WARP_SZ <= 32");
    /// @endcond

    template<typename T>
    static __device__ __forceinline__ void Add(T& value);

    /** @fn void Add(T& value, T& total)
     *  \brief warp sum
     *  @param[in] value    input value of each thread
     *  @param[out] total   total sum of all values
     *  \warning only the last thread in the WARP_SZ group has the total sum
     */
    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T& total);

    /** @fn void AddBcast(T& value, T& total)
     *  \brief warp sum
     *
     *  The result is broadcasted to all warp threads
     *  @param[in] value    input value of each thread
     *  @param[out] total   total sum of all values
     */
    template<typename T>
    static __device__ __forceinline__ void AddBcast(T& value, T& total);

    /** @fn void Add(T& value, T* total_ptr)
     *  \brief warp sum
     *
     *  @warning only LaneID equal to (WARP_SZ - 1) stores the result
     *  @param[in] value    input value of each thread
     *  @param[out] total_ptr   ptr to store the sum of all values
     */
    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T* total_ptr);

    /** @fn T AddAtom(T& value, T* total_ptr)
     *  \brief warp sum
     *
     *  Compute the warp-level prefix-sum of 'value' and add the total sum on
     *  'total_ptr' with an atomic operation.
     *  @warning only LaneID equal to (WARP_SZ - 1) stores the result
     *  @param[in] value    input value of each thread
     *  @param[out] total_ptr   ptr to store the sum of all values
     *  @return old value of total_ptr before atomicAdd operation
     */
    template<typename T>
    static __device__ __forceinline__ T AtomicAdd(T& value, T* total_ptr);

    template<typename T>
    static __device__ __forceinline__
    T AtomicAdd(T& value, T* total_ptr, T& total);

    /** @fn void Add(T* in_ptr, T* total_ptr)
     *  \brief warp sum
     *
     *  Compute the warp-level prefix-sum of the first 32 values of 'in_ptr'
     *  and store the result in same locations. The total sum is stored in
     *  'total_ptr'.
     *  @warning only LaneID equal to (WARP_SZ - 1) stores the result
     *  @param[in,out] in_ptr  input/output values
     *  @param[out] total_ptr  ptr to store the sum of all values
     */
    template<typename T>
    static __device__ __forceinline__ void Add(T* in_ptr, T* total_ptr);
};

} // namespace xlib

#include "impl/WarpScan.i.cuh"
