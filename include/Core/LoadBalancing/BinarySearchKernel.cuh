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
#include <Core/cuStingerTypes.cuh>
#include <Support/Device/BinarySearchLB.cuh>

/**
 * @brief
 */
namespace load_balacing {

/**
 * @brief
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK,
      void (*Operator)(const custinger::Vertex&, const custinger::Edge&, void*)>
__global__
void binarySearchKernel(custinger::cuStingerDevData          data,
                        const custinger::vid_t* __restrict__ d_input,
                        const int*              __restrict__ d_work,
                        int                                  work_size,
                        void*                   __restrict__ optional_field) {
    using custinger::degree_t;
    using custinger::Vertex;
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    auto lambda = [&](int pos, degree_t offset) {
                        Vertex vertex(data, d_input[pos]);
                        auto edge = vertex.edge(offset);
                        Operator(vertex, edge, optional_field);
                    };
    xlib::binarySearchLB<BLOCK_SIZE>(d_work, work_size, smem, lambda);
}

} // namespace load_balacing
