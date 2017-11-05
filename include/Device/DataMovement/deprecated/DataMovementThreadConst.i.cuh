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
namespace data_movement {
namespace thread {

namespace {
/**
* CONSTANT SIZE
*/
template<int SIZE, typename T>
__device__ __forceinline__ void SharedRegFormat(T* __restrict__ Source, T* __restrict__ Dest) {
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Dest)[i] = reinterpret_cast<int2*>(Source)[i];
    } else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(Dest)[i] = reinterpret_cast<int*>(Source)[i];
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
           reinterpret_cast<short*>(Dest)[i] = reinterpret_cast<short*>(Source)[i];
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Dest[i] = Source[i];
    }
}
} // namespace

//------------------------------------------------------------------------------

/**
* CONSTANT SIZE
*/
template<int SIZE, typename T>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    SharedRegFormat<SIZE>(Queue, SMem);
}

/**
* CONSTANT SIZE
*/
template<int SIZE, typename T>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    SharedRegFormat<SIZE>(SMem, Queue);
}

/**
* CONSTANT SIZE
*/
template<cub::CacheStoreModifier M, int SIZE, typename T>
void __device__ __forceinline__ RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ Pointer) {

    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i++)
            cub::ThreadStore<M>(reinterpret_cast<int4*>(Pointer) + i,
                                reinterpret_cast<int4*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            cub::ThreadStore<M>(reinterpret_cast<long long int*>(Pointer) + i,
                                reinterpret_cast<long long int*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            cub::ThreadStore<M>(reinterpret_cast<int*>(Pointer) + i,
                                reinterpret_cast<int*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
            cub::ThreadStore<M>(reinterpret_cast<short*>(Pointer) + i,
                                reinterpret_cast<short*>(Queue)[i]);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            cub::ThreadStore<M>(Pointer + i, Queue[i]);
    }
}

/**
* CONSTANT SIZE
*/
template<cub::CacheLoadModifier M, int SIZE, typename T>
void __device__ __forceinline__ GlobalToReg(T* __restrict__ Pointer,
                                            T (&Queue)[SIZE]) {

    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i++)
            reinterpret_cast<int4*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<int4*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<int2*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<int*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
            reinterpret_cast<short*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<short*>(Pointer) + i);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Queue[i] = cub::ThreadLoad<M>(Pointer + i);
    }
}

} //@thread
} //@data_movement
