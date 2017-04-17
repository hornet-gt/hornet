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
#include "Core/cuStinger.hpp"
#include <cstring>

namespace cu_stinger {

cuStinger::cuStinger(size_t num_vertices, size_t num_edges,
                     const off_t* csr_offset, const id_t* csr_edges) noexcept :
                           _nV(num_vertices),
                           _nE(num_edges),
                           _csr_offset(csr_offset) {

   _edge_data_ptr[0] = const_cast<byte_t*>(
                        reinterpret_cast<const byte_t*>(csr_edges));
}

cuStinger::~cuStinger() noexcept {
    cuFree(_d_nodes);
}

void cuStinger::initialize() noexcept {
    if ((NUM_EXTRA_VTYPES != 0 && !_vertex_init) ||
            (NUM_EXTRA_ETYPES != 0 && !_edge_init) || _custinger_init) {
        ERROR("Graph extra data not initialized");
    }
    _custinger_init = true;
    initializeGlobal();
    //--------------------------------------------------------------------------
    //VERTICES INITIALIZATION
    auto   h_nodes = new vertex_t[_nV];
    auto   degrees = reinterpret_cast<degree_t*>(h_nodes);
    auto    limits = degrees + _nV;
    auto ptr_array = reinterpret_cast<edge_t**>(limits + _nV);

    for (id_t i = 0; i < _nV; i++) {
        degrees[i] = _csr_offset[i + 1] - _csr_offset[i];
        limits[i]  = std::max(static_cast<id_t>(MIN_EDGES_PER_BLOCK),
                              xlib::roundup_pow2(degrees[i] + 1));
    }
    auto byte_ptr = reinterpret_cast<byte_t*>(h_nodes);
    for (int i = 0; i < NUM_EXTRA_VTYPES; i++) {
        size_t num_bytes = _nV * EXTRA_VTYPE_SIZE[i];
        std::memcpy(byte_ptr, _vertex_data_ptr[i], num_bytes);
        byte_ptr += num_bytes;
    }
    //--------------------------------------------------------------------------
    //EDGES INITIALIZATION
    for (id_t i = 0; i < _nV; i++) {
        auto          degree = degrees[i];
        const auto& mem_ptrs = mem_management.insert(degree);
        ptr_array[i]         = mem_ptrs.second;
        byte_t* h_blockarray = reinterpret_cast<byte_t*>(mem_ptrs.first);
        size_t        offset = _csr_offset[i];

        #pragma unroll
        for (int j = 0; j < NUM_ETYPES; j++) {
            size_t    num_bytes = degree * ETYPE_SIZE[j];
            size_t offset_bytes = offset * ETYPE_SIZE[j];
            std::memcpy(h_blockarray, _edge_data_ptr[j] + offset_bytes,
                        num_bytes);
        }
    }
    int num_blockarrays = mem_management.num_blockarrays();
    for (int i = 0; i < num_blockarrays; i++) {
        const auto& mem_ptrs = mem_management.get_block_array_ptr(i);
        cuMemcpyToDeviceAsync(mem_ptrs.first, EDGES_PER_BLOCKARRAY,
                              mem_ptrs.second);
    }
    cuMemcpyToDeviceAsync(h_nodes, _nV, _d_nodes);
    delete[] h_nodes;
    //mem_management.free_host_ptr();
}

} // namespace cu_stinger
