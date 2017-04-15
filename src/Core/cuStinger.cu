/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
#include "Core/cuStinger.hpp"

namespace cu_stinger {

__constant__ size_t           d_V = 0;
__constant__ VertexTypes* d_nodes = nullptr;

cuStinger::cuStinger(size_t num_vertices, size_t num_edges,
                     const off_t* csr_offset, const id_t* csr_edges) noexcept :
                           _nV(num_vertices),
                           _nE(num_edges),
                           _csr_offset(csr_offset) {

    _degrees = new degree_t[_nV];
    _limits  = new degree_t[_nV];
    for (id_t i = 0; i < _nV; i++) {
        _degrees[i] = _csr_offset[i + 1] - _csr_offset[i];
        _limits[i]  = std::max(static_cast<id_t>(MIN_EDGES_PER_BLOCK),
                               xlib::roundup_pow2(degree + 1));
    }
    _edge_data_ptr[0] = csr_edges;
}

cuStinger::~cuStinger() noexcept {
    delete[] _degrees;
    delete[] _limits;
}

void cuStinger::initialize() noexcept {
    cuMemcpyToSymbol(_nV, d_V);

    VertexTypes* nodes;
    cuMalloc(nodes, _nV);
    cuMemcpyToSymbol(nodes, d_nodes);

    auto h_nodes = new VertexTypes[_nV];

    for (id_t i = 0; i < _nV; i++) {
        const auto& mem_ptrs = mem_management.insert(_degrees[i]);

        byte_t* h_blockarray = mem_ptrs.first;

        size_t offset = _csr_offset[i];
        for (int j = 0; j < NUM_VERTEX_TYPES; j++) {
            size_t num_bytes = _degrees[i] * edge_type_sizes[j];
            std::memcpy(h_blockarray, _edge_data_ptr[j] + offset, num_bytes);
            h_blockarray += num_bytes;
        }
    }
    int num_containers = mem_management.num_blocks();
    for (int i = 0; i < num_containers; i++) {
        const auto& mem_ptrs = mem_management.get_block_array_ptr(i);
        cuMemcpyToDeviceAsync(mem_ptrs.first, EDGES_PER_BLOCKARRAY,
                              mem_ptrs.second);
    }
    //cuMemcpyToDeviceAsync(h_nodes, _d_nodes, _V);

    delete[] _degrees;
    delete[] _limits;
    _degrees = nullptr;
    _limits  = nullptr;
}

} // namespace cu_stinger
