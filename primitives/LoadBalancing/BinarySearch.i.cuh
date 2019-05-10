/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
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
#include "BinarySearchKernel.cuh"
#include "StandardAPI.hpp"
#include <Device/Primitives/CubWrapper.cuh>  //xlib::CubExclusiveSum
#include <Device/Util/DeviceProperties.cuh>  //xlib::SMemPerBlock

namespace hornets_nest {
namespace load_balancing {

template<typename HornetClass>
BinarySearch::BinarySearch(HornetClass& hornet,
                           const float work_factor) noexcept {
    //static_assert(IsHornet<HornetClass>::value,
    //             "BinarySearch: parameter is not an instance of Hornet Class");
    d_work.resize(work_factor * hornet.nV());
    prefixsum.resize(work_factor * hornet.nV());
}

inline BinarySearch::~BinarySearch() noexcept {
    //hornets_nest::gpu::free(_d_work);
}

template<typename HornetClass, typename Operator, typename vid_t>
void BinarySearch::apply(HornetClass& hornet,
                         const vid_t *      d_input,
                         int                num_vertices,
                         const Operator&    op) const noexcept {
    //static_assert(IsHornet<HornetClass>::value,
    //             "BinarySearch: paramenter is not an instance of Hornet Class");
    d_work.resize(num_vertices + 1);
    prefixsum.resize(num_vertices + 1);
    int ITEMS_PER_BLOCK = xlib::DeviceProperty
                          ::smem_per_block<vid_t>(BLOCK_SIZE);
    const auto DYN_SMEM_SIZE = ITEMS_PER_BLOCK * sizeof(vid_t);
    //assert(num_vertices < _work_size && "BinarySearch (work queue) too small");

    if (d_input != nullptr) {
    kernel::computeWorkKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (hornet.device(), d_input, num_vertices, d_work.data().get());
    } else {
    kernel::computeWorkKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (hornet.device(), num_vertices, d_work.data().get());
    }
    CHECK_CUDA_ERROR

    prefixsum.run(d_work.data().get(), num_vertices + 1);
    CHECK_CUDA_ERROR

    int total_work;
    cuMemcpyToHost(d_work.data().get() + num_vertices, total_work);
    unsigned grid_size = xlib::ceil_div(total_work, ITEMS_PER_BLOCK);

    if (total_work == 0)
        return;
    if (d_input != nullptr) {
    kernel::binarySearchKernel<BLOCK_SIZE>
        <<< grid_size, BLOCK_SIZE, DYN_SMEM_SIZE >>>
        (hornet.device(), d_input, d_work.data().get(), num_vertices + 1, op);
    } else {
    kernel::binarySearchKernel<BLOCK_SIZE>
        <<< grid_size, BLOCK_SIZE, DYN_SMEM_SIZE >>>
        (hornet.device(), d_work.data().get(), num_vertices + 1, op);
    }
    CHECK_CUDA_ERROR
}

template<typename HornetClass, typename Operator>
void BinarySearch::apply(HornetClass& hornet, const Operator& op)
                         const noexcept {
    apply<HornetClass, Operator, int>(hornet, nullptr, (int) hornet.nV(), op);
}

} // namespace load_balancing
} // namespace hornets_nest
