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

cuStingerInit::cuStingerInit(size_t num_vertices, size_t num_edges,
                             const off_t* csr_offsets, const id_t* csr_edges)
                             noexcept :
                               _nV(num_vertices),
                               _nE(num_edges),
                               _csr_offsets(csr_offsets) {

   _edge_data_ptrs[0] = const_cast<byte_t*>(
                        reinterpret_cast<const byte_t*>(csr_edges));
}

//------------------------------------------------------------------------------

cuStinger::cuStinger(const cuStingerInit& custinger_init) noexcept :
                            _nV(custinger_init._nV),
                            _nE(custinger_init._nE),
                            _csr_offsets(custinger_init._csr_offsets),
                            _vertex_data_ptrs(custinger_init._vertex_data_ptrs),
                            _edge_data_ptrs(custinger_init._edge_data_ptrs) {

    const auto lamba = [](byte_t* ptr) { return ptr != nullptr; };
    bool vertex_init_data = std::all_of(_vertex_data_ptrs,
                                   _vertex_data_ptrs + NUM_EXTRA_VTYPES, lamba);
    bool   edge_init_data = std::all_of(_edge_data_ptrs,
                                        _edge_data_ptrs + NUM_ETYPES, lamba);
    if (!vertex_init_data)
        ERROR("Vertex data not initializated");
    if (!edge_init_data)
        ERROR("Edge data not initializated");

    //--------------------------------------------------------------------------
    //////////////////////
    // COPY VERTEX DATA //
    //////////////////////
    id_t round_nV = xlib::roundup_pow2(_nV);
    cuMalloc(_d_vertices, round_nV * sizeof(vertex_t));

    for (int i = 0; i < NUM_EXTRA_VTYPES; i++) {
        byte_t* device_ptr = _d_vertices + round_nV * VTYPE_SIZE_PS[i + 1];
        cuMemcpyToDeviceAsync(_vertex_data_ptrs[i], _nV * EXTRA_VTYPE_SIZE[i],
                              device_ptr);
    }
    initializeVertexGlobal();
    //--------------------------------------------------------------------------
    ///////////////////////////////////
    // EDGES INITIALIZATION AND COPY //
    ///////////////////////////////////
    using pair_t = typename std::pair<edge_t*, degree_t>;
    auto h_vertex_basic_ptr = new pair_t[_nV];

    for (id_t i = 0; i < _nV; i++) {
        auto           degree = _csr_offsets[i + 1] - _csr_offsets[i];
        const auto&  mem_ptrs = mem_management.insert(degree);
        h_vertex_basic_ptr[i] = pair_t(mem_ptrs.second, degree);

        byte_t*  h_blockarray = reinterpret_cast<byte_t*>(mem_ptrs.first);
        size_t         offset = _csr_offsets[i];

        #pragma unroll
        for (int j = 0; j < NUM_ETYPES; j++) {
            size_t    num_bytes = degree * ETYPE_SIZE[j];
            size_t offset_bytes = offset * ETYPE_SIZE[j];
            std::memcpy(h_blockarray, _edge_data_ptrs[j] + offset_bytes,
                        num_bytes);
        }
    }
    int num_blockarrays = mem_management.num_blockarrays();
    for (int i = 0; i < num_blockarrays; i++) {
        const auto& mem_ptrs = mem_management.get_block_array_ptr(i);
        cuMemcpyToDeviceAsync(mem_ptrs.first, EDGES_PER_BLOCKARRAY,
                              mem_ptrs.second);
    }
    //--------------------------------------------------------------------------
    //////////////////////////////
    // VERTICES BASIC DATA COPY //
    //////////////////////////////
    auto d_vertex_basic_ptr = reinterpret_cast<pair_t*>(_d_vertices);
    cuMemcpyToDeviceAsync(h_vertex_basic_ptr, _nV, d_vertex_basic_ptr);
    delete[] h_vertex_basic_ptr;
    //mem_management.free_host_ptr();
}

cuStinger::~cuStinger() noexcept {
    cuFree(_d_vertices);
}

} // namespace cu_stinger



    //    limits[i]  = std::max(static_cast<id_t>(MIN_EDGES_PER_BLOCK),
    //                          xlib::roundup_pow2(degrees[i] + 1));
