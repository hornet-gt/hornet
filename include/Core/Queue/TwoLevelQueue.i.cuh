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
#include "Core/Queue/ExpandContractKernel.cuh"  //cuMemcpyToDeviceAsync
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
TwoLevelQueue<T>::TwoLevelQueue(const custinger::cuStinger& custinger,
                                bool enable_traverse,
                                size_t max_allocated_items) noexcept :
             _custinger(custinger),
             _enable_traverse(enable_traverse),
             _max_allocated_items(max_allocated_items == 0 ?
                                  custinger.nV() * 2 : max_allocated_items) {

    cuMalloc(_d_queue_ptrs.first, _max_allocated_items);
    cuMalloc(_d_queue_ptrs.second, _max_allocated_items);
    cuMalloc(_d_queue_counter, 1);
    cuMemcpyToDevice(0, _d_queue_counter);
    if (enable_traverse) {
        cuMalloc(_d_work_ptrs.first,  _max_allocated_items);
        cuMalloc(_d_work_ptrs.second, _max_allocated_items);
        cuMalloc(_d_queue2_counter, 1);
        cuMemcpyToDevice(0, const_cast<int*>(_d_work_ptrs.first));
        cuMemcpyToDevice(make_int2(0, 0), _d_queue2_counter);
    }
}

template<typename T>
TwoLevelQueue<T>::TwoLevelQueue(const TwoLevelQueue<T>& obj) noexcept :
                            _custinger(obj._custinger),
                            _max_allocated_items(obj._max_allocated_items),
                            _d_queue_ptrs(obj._d_queue_ptrs),
                            _d_queue_counter(obj._d_queue_counter),
                            _enable_delete(false) { std::cout << "copy" << std::endl; }

template<typename T>
inline TwoLevelQueue<T>::~TwoLevelQueue() noexcept {
    if (!_enable_delete)
        return;
    cuFree(_d_queue_ptrs.first, _d_queue_ptrs.second,
           _d_work_ptrs.first, _d_work_ptrs.second, _d_queue_counter);
    delete[] _host_data;
}

template<typename T>
__host__ void TwoLevelQueue<T>::insert(const T& item) noexcept {
#if defined(__CUDA_ARCH__)
    unsigned       ballot = __ballot(true);
    unsigned elected_lane = xlib::__msb(ballot);
    int warp_offset;
    if (xlib::lane_id() == elected_lane)
        warp_offset = atomicAdd(_d_queue_counter, __popc(ballot));
    int offset = __popc(ballot & xlib::LaneMaskLT()) +
                 __shfl(warp_offset, elected_lane);
    _d_queue_ptrs.second[offset] = item;
#else
    cuMemcpyToDeviceAsync(item, const_cast<int*>(_d_queue_ptrs.first) +
                                                 _num_queue_vertices);
    if (_enable_traverse)
        work_evaluate(&item, 1);
    _num_queue_vertices++;
#endif
}

template<typename T>
__host__ inline void TwoLevelQueue<T>
::insert(const T* items_array, int num_items) noexcept {
    cuMemcpyToDeviceAsync(items_array, num_items,
                          _d_queue_ptrs.first + _num_queue_vertices);
    if (_enable_traverse)
        work_evaluate(items_array, num_items);
    _num_queue_vertices += num_items;
}

template<typename T>
__host__ void TwoLevelQueue<T>::swap() noexcept {
    _d_queue_ptrs.swap();
    cuMemcpyToHostAsync(_d_queue_counter, _num_queue_vertices);
    cuMemcpyToDeviceAsync(0, _d_queue_counter);
}

template<typename T>
__host__ void TwoLevelQueue<T>::clear() noexcept {
    cuMemcpyToDevice(0, _d_queue_counter);
    if (_enable_traverse) {
        cuMemcpyToDevice(0, const_cast<int*>(_d_work_ptrs.first));
        cuMemcpyToDevice(make_int2(0, 0), _d_queue2_counter);
    }
}

template<typename T>
__host__ const T* TwoLevelQueue<T>::device_ptr_q1() const noexcept {
    return _d_queue_ptrs.first;
}

template<typename T>
__host__ const T* TwoLevelQueue<T>::device_ptr_q2() const noexcept {
    return _d_queue_ptrs.second;
}

template<typename T>
__host__ const T* TwoLevelQueue<T>::host_data() noexcept {
    if (_host_data == nullptr)
        _host_data = new T[_max_allocated_items];
    cuMemcpyToHost(_d_queue_ptrs.second, _num_queue_vertices, _host_data);
    return _host_data;
}

template<typename T>
__host__ int TwoLevelQueue<T>::size() const noexcept {
    return _num_queue_vertices;
}

template<typename T>
__host__ void TwoLevelQueue<T>::print1() noexcept {
    cuMemcpyToHost(_d_queue_counter, _num_queue_vertices);
    cu::printArray(_d_queue_ptrs.second, _num_queue_vertices);
}

template<typename T>
__host__ void TwoLevelQueue<T>::print2() noexcept {
    cuMemcpyToHost(_d_queue_counter, _num_queue_vertices);
    cu::printArray(_d_queue_ptrs.second, _num_queue_vertices);
}

//------------------------------------------------------------------------------

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
        <<< grid_size, BLOCK_SIZE >>> (_custinger.device_data(),
                                       _d_queue_ptrs,  _d_work_ptrs,
                                       _d_queue2_counter,
                                       _num_queue_vertices + 1, op);
    if (CHECK_CUDA_ERROR1)
        CHECK_CUDA_ERROR

    int2 frontier_info;
    cuMemcpyToHostAsync(_d_queue2_counter, frontier_info);
    _num_queue_vertices = frontier_info.x;
    _num_queue_edges    = frontier_info.y;
    cuMemcpyToDeviceAsync(make_int2(0, 0), _d_queue2_counter);

    cuMemcpyToDeviceAsync(_num_queue_edges, _d_work_ptrs.second +
                                            _num_queue_vertices);
    _d_queue_ptrs.swap();
    _d_work_ptrs.swap();
}

} // namespace custinger_alg
