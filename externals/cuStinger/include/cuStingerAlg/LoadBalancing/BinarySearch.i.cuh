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
#include "BinarySearchKernel.cuh"
#include "Support/Device/Definition.cuh"    //xlib::SMemPerBlock
#include "Support/Device/CubWrapper.cuh"    //xlib::CubExclusiveSum
#include "cuStingerAlg/Operator++.cuh"      //custinger::forAll

namespace load_balacing {

namespace detail {

__global__
void computeWorkKernel(const custinger::vid_t*    __restrict__ d_input,
                       const custinger::degree_t* __restrict__ d_degrees,
                       int num_vertices,
                       int* __restrict__ d_work) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < num_vertices; i += stride)
        d_work[i] = d_degrees[ d_input[i] ];
    if (id == 0)
        d_work[num_vertices] = 0;
}

} // namespace detail
//------------------------------------------------------------------------------

inline BinarySearch
::BinarySearch(const custinger::eoff_t* csr_offsets, size_t num_vertices)
    noexcept : BinarySearch(csr_offsets, num_vertices, num_vertices * 2) {}

inline BinarySearch
::BinarySearch(const custinger::eoff_t* csr_offsets, size_t num_vertices,
               int max_allocated_items) noexcept {

    cuMalloc(_d_work, max_allocated_items);
    cuMalloc(_d_degrees, num_vertices);

    auto tmp = new custinger::degree_t[num_vertices + 1];
    std::adjacent_difference(csr_offsets, csr_offsets + num_vertices, tmp);
    cuMemcpyToDevice(tmp + 1, num_vertices, _d_degrees);
    delete[] tmp;
}

inline BinarySearch::~BinarySearch() {
    cuFree(_d_work, _d_degrees);
}

template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
inline void BinarySearch::traverse_edges(const custinger::vid_t* d_input,
                                         int num_vertices,
                                         void* optional_field) noexcept {
    using custinger::vid_t;
    const int ITEMS_PER_BLOCK = xlib::SMemPerBlock<BLOCK_SIZE, vid_t>::value;

    detail::computeWorkKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (d_input, _d_degrees, num_vertices, _d_work);
        //work[num_vertices] must be zero for prefixsum*/

    if (CHECK_CUDA_ERROR1)
        CHECK_CUDA_ERROR

    xlib::CubExclusiveSum<int>(_d_work, num_vertices + 1);

    int total_work;
    cuMemcpyToHost(_d_work + num_vertices, total_work);
    unsigned grid_size = xlib::ceil_div<ITEMS_PER_BLOCK>(total_work);

    binarySearchKernel<BLOCK_SIZE, ITEMS_PER_BLOCK, Operator>
        <<< grid_size, BLOCK_SIZE >>>(d_input, _d_work, num_vertices + 1,
                                      optional_field);
    if (CHECK_CUDA_ERROR1)
        CHECK_CUDA_ERROR
}

} // namespace load_balacing
