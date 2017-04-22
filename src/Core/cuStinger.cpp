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
#include "Support/Host/Timer.hpp"//xlib::Timer
#include <cstring>               //std::memcpy

using namespace timer;

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

size_t cuStingerInit::nV() const noexcept {
    return _nV;
}

const off_t* cuStingerInit::csr_offsets() const noexcept {
    return _csr_offsets;
}

//==============================================================================

cuStinger::cuStinger(const cuStingerInit& custinger_init) noexcept :
                            _nV(custinger_init._nV),
                            _nE(custinger_init._nE) {

    auto        csr_offsets = custinger_init._csr_offsets;
    auto h_vertex_data_ptrs = custinger_init._vertex_data_ptrs;
    auto     edge_data_ptrs = custinger_init._edge_data_ptrs;

    const auto lamba = [](byte_t* ptr) { return ptr != nullptr; };
    bool vertex_init_data = std::all_of(h_vertex_data_ptrs,
                                  h_vertex_data_ptrs + NUM_EXTRA_VTYPES, lamba);
    bool   edge_init_data = std::all_of(edge_data_ptrs,
                                        edge_data_ptrs + NUM_ETYPES, lamba);
    if (!vertex_init_data)
        ERROR("Vertex data not initializated");
    if (!edge_init_data)
        ERROR("Edge data not initializated");
    Timer<DEVICE> TM;
    TM.start();
    //--------------------------------------------------------------------------
    //////////////////////
    // COPY VERTEX DATA //
    //////////////////////
    id_t round_nV = xlib::roundup_pow2(_nV);
    cuMalloc(_d_vertices, round_nV * sizeof(vertex_t));

    byte_t* d_vertex_data_ptrs[NUM_VTYPES];
    for (int i = 0; i < NUM_EXTRA_VTYPES; i++) {
        d_vertex_data_ptrs[i] = _d_vertices + round_nV * VTYPE_SIZE_PS[i + 1];
        cuMemcpyToDeviceAsync(h_vertex_data_ptrs[i], _nV * EXTRA_VTYPE_SIZE[i],
                              d_vertex_data_ptrs[i]);
    }
    initializeVertexGlobal(d_vertex_data_ptrs);
    //--------------------------------------------------------------------------
    ///////////////////////////////////
    // EDGES INITIALIZATION AND COPY //
    ///////////////////////////////////
    using pair_t = typename std::pair<edge_t*, degree_t>;
    auto h_vertex_basic_data = new pair_t[_nV];

    for (id_t i = 0; i < _nV; i++) {
        auto            degree = csr_offsets[i + 1] - csr_offsets[i];
        const auto&   mem_ptrs = mem_management.insert(degree);
        h_vertex_basic_data[i] = pair_t(mem_ptrs.second, degree);

        byte_t*   h_blockarray = reinterpret_cast<byte_t*>(mem_ptrs.first);
        size_t          offset = csr_offsets[i];

        #pragma unroll
        for (int j = 0; j < NUM_ETYPES; j++) {
            size_t    num_bytes = degree * ETYPE_SIZE[j];
            size_t offset_bytes = offset * ETYPE_SIZE[j];
            std::memcpy(h_blockarray, edge_data_ptrs[j] + offset_bytes,
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
    cuMemcpyToDeviceAsync(h_vertex_basic_data, _nV, d_vertex_basic_ptr);
    delete[] h_vertex_basic_data;
    TM.stop();
    TM.print("Initilization Time:");
    //mem_management.free_host_ptr();
}

cuStinger::~cuStinger() noexcept {
    cuFree(_d_vertices);
}

void cuStinger::check_consistency(const cuStingerInit& custinger_init)
                                    const noexcept {
    using pair_t = typename std::pair<id_t*, degree_t>;
    auto d_vertex_basic_ptr = reinterpret_cast<pair_t*>(_d_vertices);

    auto h_vertex_basic_ptr = new pair_t[_nV];
    cuMemcpyToHost(d_vertex_basic_ptr, _nV, h_vertex_basic_ptr);

    auto csr_offsets = new off_t[_nV + 1];
    csr_offsets[0] = 0;
    for (id_t i = 1; i <= _nV; i++)
        csr_offsets[i] = h_vertex_basic_ptr[i - 1].second + csr_offsets[i - 1];

    bool offset_check = std::equal(csr_offsets, csr_offsets + _nV,
                                   custinger_init._csr_offsets);
    if (!offset_check)
        ERROR("Vertex Array not consistent")
    //--------------------------------------------------------------------------
    auto csr_edges = new id_t[_nE];
    off_t   offset = 0;
    for (id_t i = 0; i < _nV; i++) {
        degree_t degree = h_vertex_basic_ptr[i].second;
        if (degree == 0) continue;
        cuMemcpyToHost(h_vertex_basic_ptr[i].first,
                       h_vertex_basic_ptr[i].second, csr_edges + offset);
        offset += degree;
    }

    auto edge_ptr = reinterpret_cast<id_t*>(custinger_init._edge_data_ptrs[0]);
    bool edges_check = std::equal(csr_edges, csr_edges + _nE, edge_ptr);
    if (!edges_check)
        ERROR("Edge Array not consistent")

    delete[] h_vertex_basic_ptr;
    delete[] csr_edges;
}

} // namespace cu_stinger
