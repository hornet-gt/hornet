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
 *
 * @file
 */
#include "BinarySearchKernel.cuh"
#include "Support/Device/Definition.cuh"    //xlib::SMemPerBlock

namespace load_balacing {

inline void work_t::swap() noexcept {
    std::swap(first, second);
}

inline BinarySearch::BinarySearch(cu_stinger_alg::TwoLevelQueue<cu_stinger::id_t>& queue,
                                  const cu_stinger::off_t* csr_offsets) noexcept : _queue(queue) {
    int max_allocated_items = queue.max_allocated_items();
    cuMalloc(_d_work.first, max_allocated_items);
    cuMalloc(_d_work.second, max_allocated_items);
    cuMemcpyToSymbol(_d_work, d_work);
    cuMemcpyToSymbol(make_int2(0, 0), d_counters);

    int queue_size = queue.size();
    auto host_data = queue.host_data();
    auto      work = new int[queue_size + 1];
    work[0] = 0;
    for (int i = 0; i < queue_size; i++)
        work[i + 1] = csr_offsets[i + 1] - csr_offsets[i];
    std::partial_sum(work + 1, work + queue_size + 1, work + 1);
    cuMemcpyToDevice(work, queue_size + 1, _d_work.first);
    _total_work = work[queue_size];
    delete[] work;
}

inline BinarySearch::~BinarySearch() {
    cuFree(_d_work.first, _d_work.second);
}

template<typename Operator, typename... TArgs>
inline void BinarySearch::traverse_edges(TArgs... optional_data) noexcept {
    int num_queue_vertices = _queue.size();

    const int ITEMS_PER_BLOCK = xlib::SMemPerBlock<BLOCK_SIZE, id_t>::value;
    int grid_size = xlib::ceil_div<ITEMS_PER_BLOCK>(_total_work);

    binarySearchKernel<ITEMS_PER_BLOCK, Operator>
        <<< grid_size, BLOCK_SIZE >>>(num_queue_vertices + 1, optional_data...);

    if (CHECK_CUDA_ERROR1)
        CHECK_CUDA_ERROR

    int2 counters;
    cuMemcpyFromSymbolAsync(d_counters, counters);
    num_queue_vertices = counters.x;
    _total_work        = counters.y;
    _queue.update_size(num_queue_vertices);

    _queue.swap();
    _d_work.swap();
    cuMemcpyToSymbolAsync(_d_work, d_work);
    cuMemcpyToDeviceAsync(_total_work, _d_work.first + num_queue_vertices);
    cuMemcpyToSymbolAsync(xlib::make2(0, 0), d_counters);        //reset counter
}

} // namespace load_balacing
