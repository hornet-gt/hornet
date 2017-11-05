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
#include "Base/Device/Primitives/WarpScan.cuh"
#include "Base/Device/Util/Basic.cuh"
#include "Base/Device/Util/Definition.cuh"

namespace xlib {

template<>
struct BlockExclusiveScan<32, SCAN_MODE::REDUCE> {
    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value, T& total) {
        const unsigned ballot = __ballot(predicate);
        value = __popc(ballot & LaneMaskLT());
        total = __popc(ballot);
    }

    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T*) {
        WarpExclusiveScan<>::AddBcast(value, total);
    }

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T*) {
        return WarpExclusiveScan<>::AtomicAdd(value, ptr, total);
    }
};

template<>
struct BlockExclusiveScan<32, SCAN_MODE::ATOMIC> :
    BlockExclusiveScan<32, SCAN_MODE::REDUCE> {};

template<unsigned BLOCK_SIZE>
struct BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::ATOMIC> {
    // smem_total is not SAFE!!!! require synchronization
    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value, T* smem_total) {
        if (threadIdx.x == 0)
            *smem_total = 0;
        __syncthreads();

        const unsigned ballot = __ballot(predicate);
        int warp_offset;
        if (lane_id() == 0)
            warp_offset = atomicAdd(smem_total, __popc(ballot));
        value = __popc(ballot & LaneMaskLT()) + __shfl(warp_offset, 0);
        // smem_total is not SAFE!!!! require synchronization
    }

    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T* shared_mem) {
        if (threadIdx.x < WARP_SIZE)
            *shared_mem = 0;
        __syncthreads();

        const T warp_offset = WarpExclusiveScan<>::AtomicAdd(value, shared_mem);
        value += warp_offset;
        __syncthreads();
        total = *shared_mem;
    }

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T* shared_mem) {
        BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::ATOMIC>
            ::AddBcast(value, total, shared_mem);

        if (threadIdx.x == 0)
            shared_mem[1] = atomicAdd(ptr, total);
        __syncthreads();
        return shared_mem[1];
    }
};

template<unsigned BLOCK_SIZE>
struct BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::REDUCE> {
    static_assert(BLOCK_SIZE != 0, "Missing template paramenter");

    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value,
                          T* smem_total, T* shared_mem) {

        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        const unsigned     ballot = __ballot(predicate);
        const unsigned   _warp_id = warp_id();

        value = __popc(ballot & LaneMaskLT());
        shared_mem[_warp_id] = __popc(ballot);
        __syncthreads();

        if (threadIdx.x < N_OF_WARPS) {
            T tmp = shared_mem[threadIdx.x];
            WarpExclusiveScan<N_OF_WARPS>::Add(tmp, smem_total);
            shared_mem[threadIdx.x] = tmp;
        }
        __syncthreads();
        value += shared_mem[_warp_id];
    }

    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T* shared_mem) {
        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        const unsigned   _warp_id = warp_id();

        WarpExclusiveScan<>::Add(value, shared_mem + _warp_id);
        __syncthreads();
        if (threadIdx.x < N_OF_WARPS) {
            T tmp = shared_mem[threadIdx.x];
            WarpExclusiveScan<N_OF_WARPS>::Add(tmp, shared_mem + N_OF_WARPS);
            shared_mem[threadIdx.x] = tmp;
        }
        __syncthreads();
        value += shared_mem[_warp_id];
        total  = shared_mem[N_OF_WARPS];
    }

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T* shared_mem) {
        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::REDUCE>
            ::AddBcast(value, total, shared_mem);

        if (threadIdx.x == 0)
            shared_mem[N_OF_WARPS + 1] = atomicAdd(ptr, total);
        __syncthreads();
        return shared_mem[N_OF_WARPS + 1];
    }
};

} // namespace xlib
