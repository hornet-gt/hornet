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
#include "../../../Util/Util.cuh"
#include "../../../../Host/BaseHost.hpp"

namespace data_movement {

using namespace PTX;
using namespace numeric;

namespace {

/**
* SMem must be in the correct position for each lane
*/
template<int SIZE, typename T>
void __device__ __forceinline__ SharedRegSupport(T* __restrict__ Source,
                                                 T* __restrict__ Dest) {

    const int SIZE_CHAR = SIZE * sizeof(T);
    if (SIZE_CHAR % 8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR / 8; i++)
            reinterpret_cast<int2*>(Dest)[i] = reinterpret_cast<int2*>(Source)[i];
    }
    else if (SIZE_CHAR % 4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR / 4; i++)
            reinterpret_cast<int*>(Dest)[i] = reinterpret_cast<int*>(Source)[i];
    }
    else if (SIZE_CHAR % 2 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR / 2; i++)
            reinterpret_cast<short*>(Dest)[i] = reinterpret_cast<short*>(Source)[i];
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR; i++)
            Dest[i] = Source[i];
    }
}
} // namespace

//==============================================================================

namespace warp_ordered {

template<cub::CacheLoadModifier M, typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    T* SMemThread = SMem + LaneID() * SIZE;

    warp::computeOffset<GLOBAL, SIZE>(SMem);
    warp::computeOffset<GLOBAL, SIZE>(Pointer);
    warp::GlobalToSharedSupport<M, SIZE * WARP_SIZE>(Pointer, SMem);

    SharedRegSupport<SIZE>(SMemThread, Queue);
}

template<cub::CacheStoreModifier M, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ Pointer) {

    T* SMemThread = SMem + LaneID() * SIZE;
    SharedRegSupport<SIZE>(Queue, SMemThread);

    warp::computeOffset<GLOBAL, SIZE>(SMem);
    warp::computeOffset<GLOBAL, SIZE>(Pointer);
    warp::SharedToGlobalSupport<M, SIZE * WARP_SIZE>(SMem, Pointer);
}

template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    SMem += LaneID() * SIZE;
    SharedRegSupport<SIZE>(Queue, SMem);
}

template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    SMem += LaneID() * SIZE;
    SharedRegSupport<SIZE>(SMem, Queue);
}

} //@warp_ordered

//------------------------------------------------------------------------------

namespace warp_ordered_adv {

template<cub::CacheLoadModifier M, typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T* __restrict__ SMemThread,
                                            T (&Queue)[SIZE]) {

    warp::GlobalToSharedSupport<M, SIZE * WARP_SIZE>(Pointer, SMem);
    SharedRegSupport<SIZE>(SMemThread, Queue);
}

template<cub::CacheStoreModifier M, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ SMemThread,
                                            T* __restrict__ Pointer) {

    SharedRegSupport<SIZE>(Queue, SMemThread);
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("%d %d %d\n", SMemThread[0], SMemThread[1], SMemThread[2]);
    warp::SharedToGlobalSupport<M, SIZE * WARP_SIZE>(SMem, Pointer);
}

template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    SharedRegSupport<SIZE>(Queue, SMem);
}

template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    SharedRegSupport<SIZE>(SMem, Queue);
}

} //@warp_ordered_advanced
} //@data_movement
