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
#include "GlobalSpace.cuh"

namespace csr {

__device__ __forceinline__
Vertex::Vertex(id_t index) {
    assert(index < d_nV);
    xlib::SeqDev<VTypeSize> VTYPE_SIZE_D;
    auto v_offsets = reinterpret_cast<off2_t*>(d_vertex_data_ptrs[0])[index];
    _degree        = v_offsets.y - v_offsets.x;
    _id            = index;
    _offset        = v_offsets.x;
    #pragma unroll
    for (int i = 0; i < NUM_EXTRA_VTYPES; i++)
        _ptrs[i] = d_vertex_data_ptrs[i + 1] + index * VTYPE_SIZE_D[i + 1];
}

__device__ __forceinline__
Edge Vertex::edge(degree_t index) const {
    return Edge(_offset + index);
}

//==============================================================================

__device__ __forceinline__
Edge::Edge(degree_t index) {
    //Edge Type Sizes Prefixsum
    xlib::SeqDev<ETypeSize> ETYPE_SIZE_D;

    _dst = reinterpret_cast<id_t*>(d_edge_data_ptrs[0])[index];
    #pragma unroll
    for (int i = 0; i < NUM_EXTRA_ETYPES; i++)
        _ptrs[i] = d_edge_data_ptrs[i + 1] + index * ETYPE_SIZE_D[i + 1];
}

} // namespace csr
