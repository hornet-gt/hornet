/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "Support/Device/PTX.cuh"
#include "Support/Device/WarpScan.cuh"
#include <cassert>

namespace custinger_alg {

template<typename T, int SIZE>
__device__ __forceinline__
DeviceQueue<T, SIZE>::DeviceQueue(T*   __restrict__ queue_ptr,
                                  int* __restrict__ size_ptr) :
                                       _queue_ptr(queue_ptr),
                                       _size_ptr(_size_ptr) {}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::insert(T item) {
    if (SIZE == 1) {
        _queue[0] = item;
        _size = 1;
    }
    else
        _queue[_size++] = item;
}

template<typename T, int SIZE>
__device__ __forceinline__
int DeviceQueue<T, SIZE>::size() const {
    return _size;
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::store() {
    if (SIZE == 1) store_ballot();
    else  store_localqueue();
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::store_localqueue() {
    assert(__ballot(true) == static_cast<unsigned>(-1));
    if (__any(_size >= SIZE)) {
        int thread_offset = _size, total;
        int   warp_offset = xlib::WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                              _size_ptr, total);
        T* ptr = _queue_ptr + warp_offset + thread_offset;
        for (int i = 0; i < _size; i++)
            ptr[i] = _queue[i];
        _size = 0;
    }
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::store_ballot() {
    unsigned       ballot = __ballot(_size);
    unsigned elected_lane = xlib::__msb(ballot);
    int warp_offset;
    if (xlib::lane_id() == elected_lane)
        warp_offset = atomicAdd(_size_ptr, __popc(ballot));
    int offset = __popc(ballot & xlib::LaneMaskLT()) +
                 __shfl(warp_offset, elected_lane);
    if (_size) {
        _queue_ptr[offset] = _queue[0];
        _size = 0;
    }
}

} // namespace custinger_alg
