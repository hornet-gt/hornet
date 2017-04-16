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

cuStinger::~cuStinger() noexcept {}

void cuStinger::initialize() noexcept {
    //--------------------------------------------------------------------------
    //VERTICES INITIALIZATION
    auto degrees = new degree_t[_nV];
    auto limits  = new degree_t[_nV];
    for (id_t i = 0; i < _nV; i++) {
        degrees[i] = _csr_offset[i + 1] - _csr_offset[i];
        limits[i]  = std::max(static_cast<id_t>(MIN_EDGES_PER_BLOCK),
                               xlib::roundup_pow2(degrees[i] + 1));
    }

    auto h_nodes = new vertex_t[_nV];
    for (int i = 0; i < NUM_VERTEX_TYPES; i++) {
        size_t num_bytes = _nV * VertexTypeSize::value[i];
        std::memcpy(h_nodes, _vertex_data_ptr[i], num_bytes);
    }
    vertex_t* d_nodes;
    cuMemcpyToDeviceAsync(h_nodes, _nV * sizeof(vertex_t), d_nodes);
    delete[] h_nodes;
    delete[] degrees;
    delete[] limits;
    //--------------------------------------------------------------------------
    //EDGES INITIALIZATION
    for (id_t i = 0; i < _nV; i++) {
        auto          degree = _degrees[i];
        const auto  mem_ptrs = mem_management.insert(degree);
        byte_t* h_blockarray = reinterpret_cast<byte_t*>(mem_ptrs.first);
        size_t        offset = _csr_offset[i];

        for (int j = 0; j < NUM_EDGE_TYPES; j++) {
            size_t num_bytes = degree * EdgeTypeSize::value[j];
            std::memcpy(h_blockarray, _edge_data_ptr[j] + offset, num_bytes);
            h_blockarray += degree * EdgeTypeSizePS::value[j];
        }
    }
    int num_containers = mem_management.num_blockarrays();
    for (int i = 0; i < num_containers; i++) {
        const auto& mem_ptrs = mem_management.get_block_array_ptr(i);
        cuMemcpyToDeviceAsync(mem_ptrs.first, EDGES_PER_BLOCKARRAY,
                              mem_ptrs.second);
    }
}

} // namespace cu_stinger
