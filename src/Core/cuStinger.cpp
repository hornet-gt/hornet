/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date July, 2017
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
#include "GraphIO/GraphBase.hpp"        //graph::structur
#include "Support/Device/Timer.cuh"     //xlib::Timer
#include "Support/Host/FileUtil.hpp"    //graph::structure
#include "Support/Host/Timer.hpp"       //xlib::Timer
#include <cstring>                      //std::memcpy

using namespace timer;

namespace custinger {

cuStingerInit::cuStingerInit(size_t num_vertices, size_t num_edges,
                             const eoff_t* csr_offsets, const vid_t* csr_edges)
                             noexcept :
                               _nV(num_vertices),
                               _nE(num_edges) {
    _vertex_data_ptrs[0] = reinterpret_cast<const byte_t*>(csr_offsets);
    _edge_data_ptrs[0]   = reinterpret_cast<const byte_t*>(csr_edges);
}

//==============================================================================

int cuStinger::global_id = 0;

cuStinger::cuStinger(const cuStingerInit& custinger_init,
                     bool traspose) noexcept :
                            _custinger_init(custinger_init),
                            _csr_offsets(_custinger_init.csr_offsets()),
                            _csr_edges(_custinger_init.csr_edges()),
                            _nV(custinger_init._nV),
                            _nE(custinger_init._nE),
                            _id(global_id++) {
    if (traspose)
        transpose();
    else
        initialize();
}

cuStinger::~cuStinger() noexcept {
    cuFree(_d_vertices, _d_csr_offsets);
    if (_internal_csr_data) {
        delete[] _csr_offsets;
        delete[] _csr_edges;
    }
    /*cuFree(d_counts, d_unique, d_degree_old, d_degree_new, d_tmp, d_ptrs_array,
            d_flags, d_inverse_pos);*/
}

void cuStinger::initialize() noexcept {
    auto edge_data_ptrs = _custinger_init._edge_data_ptrs;
    auto    vertex_data = _custinger_init._vertex_data_ptrs;
    const byte_t* h_vertex_ptrs[NUM_VTYPES];
    std::copy(vertex_data, vertex_data + NUM_VTYPES, h_vertex_ptrs);

    const auto& lamba = [](const byte_t* ptr) { return ptr != nullptr; };
    bool vertex_init_data = std::all_of(h_vertex_ptrs,
                                        h_vertex_ptrs + NUM_VTYPES, lamba);
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

    degree_t max_degree = -1;
    for (vid_t i = 0; i < _nV; i++) {
        auto degree = _csr_offsets[i + 1] - _csr_offsets[i];
        if (degree == 0) {
            h_vertex_basic_data[i] = pair_t(nullptr, 0);
            continue;
        }
        if (degree > max_degree) {
            _max_degree_vertex = i;
            max_degree = degree;
            if (degree >= EDGES_PER_BLOCKARRAY)
                ERROR("degree >= EDGES_PER_BLOCKARRAY, (", degree, ")")
        }
        const auto&   mem_data = mem_manager.insert(degree);
        h_vertex_basic_data[i] = pair_t(reinterpret_cast<edge_t*>
                                        (mem_data.second), degree);

        byte_t*   h_blockarray = mem_data.first;
        size_t          offset = _csr_offsets[i];

        #pragma unroll
        for (int j = 0; j < NUM_ETYPES; j++) {
            size_t    num_bytes = degree * ETYPE_SIZE[j];
            size_t offset_bytes = offset * ETYPE_SIZE[j];
            std::memcpy(h_blockarray + EDGES_PER_BLOCKARRAY * ETYPE_SIZE_PS[j],
                        edge_data_ptrs[j] + offset_bytes, num_bytes);
        }
    }

    //copy BlockArrays to the device
    int num_blockarrays = mem_manager.num_blockarrays();
    for (int i = 0; i < num_blockarrays; i++) {
        const auto& mem_data = mem_manager.get_blockarray_ptr(i);
        cuMemcpyToDeviceAsync(mem_data.first,
                              EDGES_PER_BLOCKARRAY * sizeof(edge_t),
                              mem_data.second);
    }
    //--------------------------------------------------------------------------
    ////////////////////////
    // COPY VERTICES DATA //
    ////////////////////////
    vid_t round_nV = xlib::roundup_pow2(_nV);
    cuMalloc(_d_vertices, round_nV * sizeof(vertex_t));

    h_vertex_ptrs[0]  = reinterpret_cast<byte_t*>(h_vertex_basic_data);
    _d_vertex_ptrs[0] = reinterpret_cast<byte_t*>(_d_vertices);
    for (int i = 0; i < NUM_VTYPES; i++) {
        _d_vertex_ptrs[i] = _d_vertices + round_nV * VTYPE_SIZE_PS[i];
        cuMemcpyToDeviceAsync(h_vertex_ptrs[i], _nV * VTYPE_SIZE[i],
                              _d_vertex_ptrs[i]);
    }
    delete[] h_vertex_basic_data;

    TM.stop();
    TM.print("Initilization Time:");
    //mem_manager.free_host_ptr();
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

    auto offsets_check = std::equal(csr_offsets, csr_offsets + _nV,
                                    custinger_init.csr_offsets());
    if (!offsets_check)
        ERROR("Vertex Array not consistent")
    auto edge_ref = custinger_init._edge_data_ptrs[0];
    auto neighbor_ptr = reinterpret_cast<const vid_t*>(edge_ref);
    if (!std::equal(csr_edges, csr_edges + _nE, neighbor_ptr))
        ERROR("Edge Array not consistent")
    delete[] csr_offsets;
    delete[] csr_edges;
}

void cuStinger::store_snapshot(const std::string& filename) const noexcept {
    auto csr_offsets = new eoff_t[_nV + 1];
    auto csr_edges   = new vid_t[_nE];
    convert_to_csr(csr_offsets, csr_edges);

    graph::StructureProp structure(graph::structure_prop::DIRECTED);
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

} // namespace custinger
