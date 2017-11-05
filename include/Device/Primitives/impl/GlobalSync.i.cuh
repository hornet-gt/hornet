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

#include "Device/DataMovement/CacheModifier.cuh"

namespace xlib {

template<unsigned BLOCK_SIZE>
__device__ __forceinline__ void globalSync() {
    //static_assert(RESIDENT_BLOCKS<BLOCK_SIZE>::value <= BLOCK_SIZE,
    //              PRINT_ERR("globalSync : RESIDENT_BLOCKS > BLOCK_SIZE"));

    volatile unsigned* VolatilePtr = GlobalSyncArray;
    __syncthreads(); //-> all warps conclude their work

    if (blockIdx.x == 0) {
        if (threadIdx.x == 0)
            VolatilePtr[blockIdx.x] = 1;

        if (threadIdx.x < gridDim.x)
            while ( Load<CG>(GlobalSyncArray + threadIdx.x) == 0 );

        __syncthreads();

        if (threadIdx.x < gridDim.x)
            Store<CG>(GlobalSyncArray + threadIdx.x, 0u);
    }
    else {
        if (threadIdx.x == 0) {
            VolatilePtr[blockIdx.x] = 1;
            while (Load<CG>(GlobalSyncArray + blockIdx.x) == 1);
        }
        __syncthreads();
    }
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__ void globalSync_v2() {
    volatile unsigned* VolatilePtr = GlobalSyncArray;
    __syncthreads(); //-> all warps conclude their work

    if (blockIdx.x == 0) {
        if (threadIdx.x == 0)
            VolatilePtr[blockIdx.x] = 1;


        for (unsigned id = threadIdx.x; id < gridDim.x; id += BLOCK_SIZE)
            while ( Load<CG>(GlobalSyncArray + id) == 0 );

        __syncthreads();

        for (unsigned id = threadIdx.x; id < gridDim.x; id += BLOCK_SIZE)
            Store<CG>(GlobalSyncArray + id, 0u);
    }
    else {
        if (threadIdx.x == 0) {
            VolatilePtr[blockIdx.x] = 1;
            while (Load<CG>(GlobalSyncArray + blockIdx.x) == 1);
        }
        __syncthreads();
    }
}

} // namespace xlib
