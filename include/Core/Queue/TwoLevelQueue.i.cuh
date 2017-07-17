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
//#include "Core/Queue/ExpandContractKernel.cuh"  //cuMemcpyToDeviceAsync
#include <Support/Device/Definition.cuh>        //xlib::SMemPerBlock
#include <Support/Device/PrintExt.cuh>          //cu::printArray
#include <Support/Device/SafeCudaAPI.cuh>       //cuMemcpyToDeviceAsync

namespace custinger_alg {

template<typename T>
inline void ptr2_t<T>::swap() noexcept {
    std::swap(const_cast<T*&>(first), second);
}

//------------------------------------------------------------------------------

template<typename T>
TwoLevelQueue<T>::TwoLevelQueue(const custinger::cuStinger& custinger)
                               noexcept :
                                  _max_allocated_items(custinger.nV() * 2) {

    cuMalloc(_d_queue_ptrs.first, _max_allocated_items);
    cuMalloc(_d_queue_ptrs.second, _max_allocated_items);
    cuMalloc(_d_counters, 1);
    cuMemset0x00(_d_counters);
    /*if (enable_traverse) {
        cuMalloc(_d_work_ptrs.first,  _max_allocated_items);
        cuMalloc(_d_work_ptrs.second, _max_allocated_items);
        cuMalloc(_d_counters, 1);
        cuMemcpyToDevice(0, const_cast<int*>(_d_work_ptrs.first));
        cuMemcpyToDevice(make_int2(0, 0), _d_counters);
    }*/
}

template<typename T>
TwoLevelQueue<T>::TwoLevelQueue(const TwoLevelQueue<T>& obj) noexcept :
                            _max_allocated_items(obj._max_allocated_items),
                            _d_queue_ptrs(obj._d_queue_ptrs),
                            _d_counters(obj._d_counters),
                            _kernel_copy(true) {
}

template<typename T>
inline TwoLevelQueue<T>::~TwoLevelQueue() noexcept {
    /*if (!_enable_delete)
        return;
    cuFree(_d_queue_ptrs.first, _d_queue_ptrs.second,
           _d_work_ptrs.first, _d_work_ptrs.second, _d_queue_counter);
    delete[] _host_data;*/
    if (!_kernel_copy)
        cuFree(_d_queue_ptrs.first, _d_queue_ptrs.second, _d_counters);
}
/*
template<typename T>
inline TwoLevelQueue<T>::init(size_t size) noexcept {
    cuMalloc(_d_queue_ptrs.first, _max_allocated_items);
    cuMalloc(_d_queue_ptrs.second, _max_allocated_items);
    cuMalloc(_d_queue_counter, 1);
    cuMemcpyToDevice(0, _d_queue_counter);
    if (enable_traverse) {
        cuMalloc(_d_work_ptrs.first,  _max_allocated_items);
        cuMalloc(_d_work_ptrs.second, _max_allocated_items);
        cuMalloc(_d_counters, 1);
        cuMemcpyToDevice(0, const_cast<int*>(_d_work_ptrs.first));
        cuMemcpyToDevice(make_int2(0, 0), _d_counters);
    }
}*/

template<typename T>
__host__ void TwoLevelQueue<T>::insert(const T& item) noexcept {
#if defined(__CUDA_ARCH__)
    unsigned       ballot = __ballot(true);
    unsigned elected_lane = xlib::__msb(ballot);
    int warp_offset;
    if (xlib::lane_id() == elected_lane)
        warp_offset = atomicAdd(&_d_counters->y, __popc(ballot));
        //warp_offset = atomicAdd(_d_queue_counter, __popc(ballot));
    int offset = __popc(ballot & xlib::LaneMaskLT()) +
                 __shfl(warp_offset, elected_lane);
    _d_queue_ptrs.second[offset] = item;
#else
    cuMemcpyToHost(_d_counters, _h_counters);
    cuMemcpyToDevice(item, const_cast<int*>(_d_queue_ptrs.first) +
                                            _h_counters.x);
    _h_counters.x++;
    cuMemcpyToDevice(_h_counters, _d_counters);
#endif
}

template<typename T>
__host__ inline
void TwoLevelQueue<T>::insert(const T* items_array, int num_items) noexcept {
    cuMemcpyToHost(_d_counters, _h_counters);
    cuMemcpyToDevice(items_array, num_items,
                     _d_queue_ptrs.first + _h_counters.x);
    _h_counters.x += num_items;
    cuMemcpyToDevice(_h_counters, _d_counters);
}

template<typename T>
__host__ void TwoLevelQueue<T>::swap() noexcept {
    //ptr2_t<T> h_pointers;
    //cuMemcpyToHost(_d_queue_ptrs, h_pointers);
    _d_queue_ptrs.swap();
    //cuMemcpyToDevice(h_pointers, _d_queue_ptrs);

    cuMemcpyToHost(_d_counters, _h_counters);
    _h_counters.x = _h_counters.y;
    _h_counters.y = 0;
    cuMemcpyToDevice(_h_counters, _d_counters);
    //cuMemcpyToHostAsync(_d_queue_counter, _num_queue_vertices);
    //cuMemcpyToDeviceAsync(0, _d_queue_counter);
    //cuMemset0x00(_d_queue_counter);
}

template<typename T>
__host__ void TwoLevelQueue<T>::clear() noexcept {
    cuMemset0x00(_d_counters);
    //cuMemset0x00(_d_queue_counter);
    /*if (_enable_traverse) {
        cuMemcpyToDevice(0, const_cast<int*>(_d_work_ptrs.first));
        cuMemcpyToDevice(make_int2(0, 0), _d_counters);
    }*/
}

template<typename T>
__host__ const T* TwoLevelQueue<T>::device_input_ptr() const noexcept {
    return _d_queue_ptrs.first;
}

template<typename T>
__host__ const T* TwoLevelQueue<T>::device_output_ptr() const noexcept {
    return _d_queue_ptrs.second;
}
/*
template<typename T>
__host__ const T* TwoLevelQueue<T>::host_data() noexcept {
    if (_host_data == nullptr)
        _host_data = new T[_max_allocated_items];
    cuMemcpyToHost(_d_queue_ptrs.second, _num_queue_vertices, _host_data);
    return _host_data;
}*/
/*
template<typename T>
__host__ int TwoLevelQueue<T>::size() noexcept {
    cuMemcpyToHost(_d_queue_counter, _num_queue_vertices);
    return _num_queue_vertices;
}*/

template<typename T>
__host__ int TwoLevelQueue<T>::size() noexcept {
    int2 _h_counters;
    cuMemcpyToHost(_d_counters, _h_counters);
    return _h_counters.x;
}

template<typename T>
__host__ int TwoLevelQueue<T>::output_size() noexcept {
    int2 _h_counters;
    cuMemcpyToHost(_d_counters, _h_counters);
    return _h_counters.y;
}

template<typename T>
__host__ void TwoLevelQueue<T>::print_input() noexcept {
    int2 _h_counters;
    cuMemcpyToHost(_d_counters, _h_counters);
    cu::printArray(_d_queue_ptrs.first, _h_counters.x);
}

template<typename T>
__host__ void TwoLevelQueue<T>::print_output() noexcept {
    int2 _h_counters;
    cuMemcpyToHost(_d_counters, _h_counters);
    cu::printArray(_d_queue_ptrs.second, _h_counters.y);
}

//------------------------------------------------------------------------------
/*
template<typename T>
__host__ void
TwoLevelQueue<T>
::work_evaluate(const custinger::vid_t* items_array, int num_items) noexcept {
    using custinger::vid_t;

    auto work = new int[num_items + 1];
    work[0] = _num_queue_edges;
    for (int i = 0; i < num_items; i++) {
        vid_t index = items_array[i];
        work[i + 1] = _custinger.csr_offsets()[index + 1] -
                      _custinger.csr_offsets()[index];
    }
    std::partial_sum(work, work + num_items + 1, work);
    auto ptr = const_cast<int*>(_d_work_ptrs.first) + _num_queue_vertices;
    cuMemcpyToDevice(work, num_items + 1, ptr);
    _num_queue_edges = work[num_items];
    delete[] work;
}


template<typename T>
template<typename Operator>
__host__ void TwoLevelQueue<T>::traverse_edges(Operator op) noexcept {
    static_assert(sizeof(T) != sizeof(T),
              "\nTwoLevelQueue::traverse_edges() is disabled for T != vid_t\n");
}

template<>
template<typename Operator>
__host__ void TwoLevelQueue<custinger::vid_t>
::traverse_edges(Operator op) noexcept {
    using custinger::vid_t;
    const int ITEMS_PER_BLOCK = xlib::SMemPerBlock<BLOCK_SIZE, vid_t>::value;
    if (!_enable_traverse)
        ERROR("traverse_edges() not enabled: wrong costructor");

    int grid_size = xlib::ceil_div<ITEMS_PER_BLOCK>(_num_queue_edges);
    if (PRINT_VERTEX_FRONTIER)
        cu::printArray(_d_queue_ptrs.first, _num_queue_vertices);

    ExpandContractLBKernel<BLOCK_SIZE, ITEMS_PER_BLOCK>
        <<< grid_size, BLOCK_SIZE >>> (_custinger.device_side(),
                                       _d_queue_ptrs, _d_work_ptrs,
                                       _d_counters,
                                       _num_queue_vertices + 1, op);
    if (CHECK_CUDA_ERROR1)
        CHECK_CUDA_ERROR

    int2 frontier_info;
    cuMemcpyToHostAsync(_d_counters, frontier_info);
    _num_queue_vertices = frontier_info.x;
    _num_queue_edges    = frontier_info.y;
    cuMemcpyToDeviceAsync(make_int2(0, 0), _d_counters);

    cuMemcpyToDeviceAsync(_num_queue_edges, _d_work_ptrs.second +
                                            _num_queue_vertices);
    _d_queue_ptrs.swap();
    _d_work_ptrs.swap();
}*/

} // namespace custinger_alg
