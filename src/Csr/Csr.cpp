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
#include "Csr/Csr.hpp"
#include "Support/Device/SafeCudaAPI.cuh"   //cuMalloc, cuMemcpyToDevice
#include "Support/Host/Numeric.hpp"         //xlib::upper_approx
#include "Support/Host/Timer.hpp"           //timer::Timer

using namespace timer;

namespace csr {

Csr::Csr(const cu_stinger::cuStingerInit& custinger_init) noexcept :
                            _nV(custinger_init._nV),
                            _nE(custinger_init._nE) {

    auto      csr_offsets = custinger_init.csr_offsets();
    auto h_edge_data_ptrs = custinger_init._edge_data_ptrs;
    auto      vertex_data = custinger_init._vertex_data_ptrs;
    const byte_t* h_vertex_data_ptrs[NUM_VTYPES];
    std::copy(vertex_data, vertex_data + NUM_VTYPES, h_vertex_data_ptrs);

    const auto lamba = [](const byte_t* ptr) { return ptr != nullptr; };
    bool vertex_init_data = std::all_of(h_vertex_data_ptrs,
                                        h_vertex_data_ptrs + NUM_VTYPES, lamba);
    bool   edge_init_data = std::all_of(h_edge_data_ptrs,
                                        h_edge_data_ptrs + NUM_ETYPES, lamba);
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
    using      pair_t = std::pair<id_t, id_t>;
    auto csr2_offsets = new pair_t[_nV];
    for (id_t i = 0; i < _nV; i++)
        csr2_offsets[i] = pair_t(csr_offsets[i], csr_offsets[i + 1]);

    //see cudaMallocPitch
    id_t round_nV = xlib::upper_approx(_nV, 512 / sizeof(vertex_t));
    cuMalloc(_d_vertices, round_nV * sizeof(vertex_t));

    byte_t* d_vertex_data_ptrs[NUM_VTYPES];
    h_vertex_data_ptrs[0] = reinterpret_cast<byte_t*>(csr2_offsets);
    for (int i = 0; i < NUM_VTYPES; i++) {
        d_vertex_data_ptrs[i] = _d_vertices + round_nV * VTYPE_SIZE_PS[i];
        cuMemcpyToDeviceAsync(h_vertex_data_ptrs[i], _nV * VTYPE_SIZE[i],
                              d_vertex_data_ptrs[i]);
    }
    delete[] csr2_offsets;
    //--------------------------------------------------------------------------
    ////////////////
    // EDGES COPY //
    ////////////////
    off_t round_nE = xlib::upper_approx(_nE, 512 / sizeof(edge_t));
    cuMalloc(_d_edges, round_nE * sizeof(edge_t));

    byte_t* d_edge_data_ptrs[NUM_ETYPES];
    for (int i = 0; i < NUM_ETYPES; i++) {
        d_edge_data_ptrs[i] = _d_edges + round_nE * ETYPE_SIZE_PS[i];
        cuMemcpyToDeviceAsync(h_edge_data_ptrs[i], _nE * ETYPE_SIZE[i],
                              d_edge_data_ptrs[i]);
    }
    initializeGlobal(d_vertex_data_ptrs, d_edge_data_ptrs);

    TM.stop();
    TM.print("Initilization Time:");
}

Csr::~Csr() noexcept {
    cuFree(_d_vertices, _d_edges);
}

} // namespace csr
