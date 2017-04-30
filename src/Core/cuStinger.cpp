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
#include "GraphIO/GraphBase.hpp"        //graph::structure
#include "Support/Host/FileUtil.hpp"    //graph::structure
#include "Support/Host/Timer.hpp"       //xlib::Timer
#include <cstring>                      //std::memcpy

using namespace timer;

namespace cu_stinger {

cuStingerInit::cuStingerInit(size_t num_vertices, size_t num_edges,
                             const eoff_t* csr_offsets, const vid_t* csr_edges)
                             noexcept :
                               _nV(num_vertices),
                               _nE(num_edges) {
    _vertex_data_ptrs[0] = reinterpret_cast<const byte_t*>(csr_offsets);
   _edge_data_ptrs[0]    = reinterpret_cast<const byte_t*>(csr_edges);
}

size_t cuStingerInit::nV() const noexcept {
    return _nV;
}

size_t cuStingerInit::nE() const noexcept {
    return _nE;
}

const eoff_t* cuStingerInit::csr_offsets() const noexcept {
    return reinterpret_cast<const eoff_t*>(_vertex_data_ptrs[0]);
}

const vid_t* cuStingerInit::csr_edges() const noexcept {
    return reinterpret_cast<const vid_t*>(_edge_data_ptrs[0]);
}

//==============================================================================

cuStinger::cuStinger(const cuStingerInit& custinger_init) noexcept :
                            _nV(custinger_init._nV),
                            _nE(custinger_init._nE) {

    auto    csr_offsets = custinger_init.csr_offsets();
    auto edge_data_ptrs = custinger_init._edge_data_ptrs;
    auto    vertex_data = custinger_init._vertex_data_ptrs;
    const byte_t* h_vertex_data_ptrs[NUM_VTYPES];
    std::copy(vertex_data, vertex_data + NUM_VTYPES, h_vertex_data_ptrs);

    const auto lamba = [](const byte_t* ptr) { return ptr != nullptr; };
    bool vertex_init_data = std::all_of(h_vertex_data_ptrs,
                                        h_vertex_data_ptrs + NUM_VTYPES, lamba);
    bool   edge_init_data = std::all_of(edge_data_ptrs,
                                        edge_data_ptrs + NUM_ETYPES, lamba);
    if (!vertex_init_data)
        ERROR("Vertex data not initializated");
    if (!edge_init_data)
        ERROR("Edge data not initializated");
    Timer<DEVICE> TM;
    TM.start();
    //--------------------------------------------------------------------------
    ///////////////////////////////////
    // EDGES INITIALIZATION AND COPY //
    ///////////////////////////////////
    using pair_t = typename std::pair<edge_t*, degree_t>;
    auto h_vertex_basic_data = new pair_t[_nV];

    for (vid_t i = 0; i < _nV; i++) {
        auto degree = csr_offsets[i + 1] - csr_offsets[i];
        if (degree == 0) {
            h_vertex_basic_data[i] = pair_t(nullptr, 0);
            continue;
        }
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
    //copy BlockArrays to the device
    int num_blockarrays = mem_management.num_blockarrays();
    for (int i = 0; i < num_blockarrays; i++) {
        const auto& mem_ptrs = mem_management.get_block_array_ptr(i);
        cuMemcpyToDeviceAsync(mem_ptrs.first, EDGES_PER_BLOCKARRAY,
                              mem_ptrs.second);
    }
    //--------------------------------------------------------------------------
    ////////////////////////
    // COPY VERTICES DATA //
    ////////////////////////
    vid_t round_nV = xlib::roundup_pow2(_nV);
    cuMalloc(_d_vertices, round_nV * sizeof(vertex_t));

    byte_t* d_vertex_data_ptrs[NUM_VTYPES];
    h_vertex_data_ptrs[0] = reinterpret_cast<byte_t*>(h_vertex_basic_data);
    d_vertex_data_ptrs[0] = reinterpret_cast<byte_t*>(_d_vertices);
    for (int i = 0; i < NUM_VTYPES; i++) {
        d_vertex_data_ptrs[i] = _d_vertices + round_nV * VTYPE_SIZE_PS[i];
        cuMemcpyToDeviceAsync(h_vertex_data_ptrs[i], _nV * VTYPE_SIZE[i],
                              d_vertex_data_ptrs[i]);
    }
    initializeVertexGlobal(d_vertex_data_ptrs);
    delete[] h_vertex_basic_data;

    TM.stop();
    TM.print("Initilization Time:");
    //mem_management.free_host_ptr();
}

cuStinger::~cuStinger() noexcept {
    cuFree(_d_vertices);
}

void cuStinger::convert_to_csr(eoff_t* csr_offsets, vid_t* csr_edges)
                               const noexcept {

    using pair_t = typename std::pair<vid_t*, degree_t>;
    auto d_vertex_basic_ptr = reinterpret_cast<pair_t*>(_d_vertices);

    auto h_vertex_basic_ptr = new pair_t[_nV];
    cuMemcpyToHost(d_vertex_basic_ptr, _nV, h_vertex_basic_ptr);

    csr_offsets[0] = 0;
    for (vid_t i = 1; i <= _nV; i++)
        csr_offsets[i] = h_vertex_basic_ptr[i - 1].second + csr_offsets[i - 1];
    //--------------------------------------------------------------------------
    eoff_t   offset = 0;
    for (vid_t i = 0; i < _nV; i++) {
        degree_t degree = h_vertex_basic_ptr[i].second;
        if (degree == 0) continue;
        cuMemcpyToHostAsync(h_vertex_basic_ptr[i].first,
                            h_vertex_basic_ptr[i].second, csr_edges + offset);
        offset += degree;
    }
    cudaDeviceSynchronize();
    delete[] h_vertex_basic_ptr;
}

void cuStinger::check_consistency(const cuStingerInit& custinger_init)
                                  const noexcept {
    auto csr_offsets = new eoff_t[_nV + 1];
    auto csr_edges   = new vid_t[_nE];
    convert_to_csr(csr_offsets, csr_edges);

    auto offset_check = std::equal(csr_offsets, csr_offsets + _nV,
                                   custinger_init.csr_offsets());
    if (!offset_check)
        ERROR("Vertex Array not consistent")
    auto edge_ref = custinger_init._edge_data_ptrs[0];
    auto edge_ptr = reinterpret_cast<const vid_t*>(edge_ref);
    if (!std::equal(csr_edges, csr_edges + _nE, edge_ptr))
        ERROR("Edge Array not consistent")
    delete[] csr_offsets;
    delete[] csr_edges;
}

void cuStinger::store_snapshot(const std::string& filename) const noexcept {
    auto csr_offsets = new eoff_t[_nV + 1];
    auto csr_edges   = new vid_t[_nE];
    convert_to_csr(csr_offsets, csr_edges);

    graph::Structure structure(graph::Structure::DIRECTED);
    size_t  base_size = sizeof(_nV) + sizeof(_nE) + sizeof(structure);
    size_t file_size1 = (static_cast<size_t>(_nV) + 1) * sizeof(eoff_t) +
                        (static_cast<size_t>(_nE)) * sizeof(vid_t);

    size_t file_size  = base_size + file_size1;

    std::cout << "Graph To binary file: " << filename
              << " (" << (file_size >> 20) << ") MB" << std::endl;

    std::string class_id = xlib::type_name<vid_t>() + xlib::type_name<eoff_t>();
    file_size           += class_id.size();
    xlib::MemoryMapped memory_mapped(filename.c_str(), file_size,
                                     xlib::MemoryMapped::WRITE, true);

    memory_mapped.write(class_id.c_str(), class_id.size(),              //NOLINT
                        &_nV, 1, &_nE, 1,                               //NOLINT
                        csr_offsets, _nV + 1, csr_edges, _nE);          //NOLINT

    delete[] csr_offsets;
    delete[] csr_edges;
}

} // namespace cu_stinger
