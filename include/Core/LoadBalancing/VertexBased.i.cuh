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
#include "VertexBasedKernel.cuh"
#include "Device/Definition.cuh"    //xlib::SMemPerBlock
#include "Device/CubWrapper.cuh"    //xlib::CubExclusiveSum
//#include "cuStingerAlg/Operator++.cuh"      //custinger::forAll

namespace load_balacing {

template<typename Operator>
void VertexBased::apply(custinger::cuStinger& custinger,
                        const custinger::vid_t* d_input, int num_vertices,
                        const Operator& op) noexcept {

    detail::vertexBasedKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (custinger, d_input, num_vertices, op);
}

template<typename Operator>
void VertexBased::apply(custinger::cuStinger& custinger,
                        const Operator& op) noexcept {

    detail::vertexBasedKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (custinger, op);
}

/*
template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
inline void VertexBased::traverse_edges(const custinger::vid_t* d_input,
                                         int num_vertices,
                                         void* optional_field) noexcept {
    @details::VertexBasedKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (d_input, num_verticesk, optional_field);

    if (CHECK_CUDA_ERROR1)
        CHECK_CUDA_ERROR
}

template<typename Operator>
inline void VertexBased::traverse_edges(const custinger::vid_t* d_input,
                                        int num_vertices,
                                        Operator op) noexcept {
    @details::VertexBasedKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (d_input, num_vertices, op);

    if (CHECK_CUDA_ERROR1)
        CHECK_CUDA_ERROR
}*/

} // namespace load_balacing
