/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date October, 2017
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
#include <Device/Definition.cuh>        //xlib::SMemPerBlock
#include <Device/PrintExt.cuh>          //cu::printArray
#include <Device/PTX.cuh>              //xlib::__msb
#include <Device/SafeCudaAPI.cuh>       //cuMemcpyToDeviceAsync
#include <BasicTypes.hpp>

namespace hornets_nest {

template<typename T>
void ptr2_t<T>::swap() noexcept {
    std::swap(const_cast<T*&>(first), second);
}

//------------------------------------------------------------------------------

template<typename T>
template<typename HornetClass>
TwoLevelQueue<T>::TwoLevelQueue(const HornetClass& hornet,
                                const float work_factor) noexcept :
                              _max_allocated_items(hornet.nV() * work_factor) {
    static_assert(IsHornet<HornetClass>::value,
                 "TwoLevelQueue paramenter is not an instance of Hornet Class");
    cuMalloc(_d_queue_ptrs.first, _max_allocated_items);
    cuMalloc(_d_queue_ptrs.second, _max_allocated_items);
    cuMalloc(_d_counters, 1);
    cuMemset0x00(_d_counters);
}

template<typename T>
TwoLevelQueue<T>::TwoLevelQueue(const TwoLevelQueue<T>& obj) noexcept :
                            _max_allocated_items(obj._max_allocated_items),
                            _d_queue_ptrs(obj._d_queue_ptrs),
                            _d_counters(obj._d_counters),
                            _h_counters(obj._h_counters),
                            _kernel_copy(true) {}

template<typename T>
TwoLevelQueue<T>::~TwoLevelQueue() noexcept {
    if (!_kernel_copy)
        cuFree(_d_queue_ptrs.first, _d_queue_ptrs.second, _d_counters);
}

template<typename T>
template<typename HornetClass>
void TwoLevelQueue<T>::initialize(const HornetClass& hornet,
                                 const float work_factors) noexcept {
    static_assert(IsHornet<HornetClass>::value,
                 "TwoLevelQueue paramenter is not an instance of Hornet Class");
    cuMalloc(_d_queue_ptrs.first, _max_allocated_items);
    cuMalloc(_d_queue_ptrs.second, _max_allocated_items);
    cuMalloc(_d_counters, 1);
    cuMemset0x00(_d_counters);
}

//------------------------------------------------------------------------------

template<typename T>
__host__ __device__ __forceinline__
void TwoLevelQueue<T>::insert(const T& item) noexcept {
#if defined(__CUDA_ARCH__)
    unsigned       ballot = __activemask();
    unsigned elected_lane = xlib::__msb(ballot);
    int warp_offset;
    if (xlib::lane_id() == elected_lane)
        warp_offset = atomicAdd(&_d_counters->y, __popc(ballot));
    int offset = __popc(ballot & xlib::lanemask_lt()) +
                 __shfl_sync(0xFFFFFFFF, warp_offset, elected_lane);
    _d_queue_ptrs.second[offset] = item;
#else
    cuMemcpyToHost(_d_counters, _h_counters);
    cuMemcpyToDevice(item, const_cast<int*>(_d_queue_ptrs.first) +
                                            _h_counters.x);
    _h_counters.x++;
    _enqueue_items++;
    cuMemcpyToDevice(_h_counters, _d_counters);
#endif
}

template<typename T>
void TwoLevelQueue<T>::insert(const T* items_array, int num_items) noexcept {
    cuMemcpyToHost(_d_counters, _h_counters);
    cuMemcpyToDevice(items_array, num_items,
                     _d_queue_ptrs.first + _h_counters.x);
    _h_counters.x  += num_items;
    _enqueue_items += num_items;
    cuMemcpyToDevice(_h_counters, _d_counters);
}

template<typename = void>
__global__ void swapKernel(int2* d_counters) {
    auto counters = *d_counters;
    counters.x    = counters.y;
    counters.y    = 0;
    *d_counters   = counters;
}

template<typename T>
void TwoLevelQueue<T>::sync() const noexcept {
    cuMemcpyToHostAsync(_d_counters, _h_counters);
    assert(_h_counters.x < _max_allocated_items && "TwoLevelQueue too small");
    assert(_h_counters.y < _max_allocated_items && "TwoLevelQueue too small");
}

template<typename T>
void TwoLevelQueue<T>::swap() noexcept {
    _d_queue_ptrs.swap();
    swapKernel<<< 1, 1 >>>(_d_counters);
    sync();
    _enqueue_items += _h_counters.x;
}

template<typename T>
void TwoLevelQueue<T>::clear() noexcept {
    _enqueue_items = 0;
    _h_counters    = { 0, 0 };
    cuMemset0x00(_d_counters);
}

template<typename T>
const T* TwoLevelQueue<T>::device_input_ptr() const noexcept {
    return _d_queue_ptrs.first;
}

template<typename T>
const T* TwoLevelQueue<T>::device_output_ptr() const noexcept {
    return _d_queue_ptrs.second;
}

template<typename T>
int TwoLevelQueue<T>::size() const noexcept {
    return _h_counters.x;
}

template<typename T>
int TwoLevelQueue<T>::output_size() const noexcept {
    return _h_counters.y;
}

template<typename T>
void TwoLevelQueue<T>::print() const noexcept {
    sync();
    cu::printArray(_d_queue_ptrs.first, _h_counters.x);
}

template<typename T>
void TwoLevelQueue<T>::print_output() const noexcept {
    sync();
    cu::printArray(_d_queue_ptrs.second, _h_counters.y);
}

template<typename T>
int TwoLevelQueue<T>::enqueue_items() const noexcept {
    return _enqueue_items;
}

} // namespace hornets_nest
