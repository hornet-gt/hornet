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
#include "Base/Device/DataMovement/DataMovementDynamic.cuh"

namespace data_movement {
namespace Queue {

template<MODE QM,
         cub::CacheStoreModifier M,
         typename T, typename R, int SIZE>
__device__ __forceinline__
void Store(T (&Queue)[SIZE],
           const int size,
           T* __restrict__ queue_ptr,
           R* __restrict__ queue_size_ptr) {

    using namespace primitives;
    using namespace data_movement::dynamic;
    int th_offset = size;
    int warp_offset = WarpExclusiveScan<>::AddAtom(th_offset, queue_size_ptr);
    if (QM == MODE::SIMPLE) {
        thread::RegToGlobal_Simple<M>(Queue, size,
                                   queue_ptr + warp_offset + th_offset);
    } else if (QM == MODE::UNROLL) {
        thread::RegToGlobal_Unroll<M>(Queue, size,
                                   queue_ptr + warp_offset + th_offset);
    }
}

} //@Queue
} //@data_movement
