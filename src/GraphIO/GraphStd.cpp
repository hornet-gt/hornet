/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
#include "GraphIO/GraphStd.hpp"
#include "Support/Host/Basic.hpp"     //ERROR
#include "Support/Host/FileUtil.hpp"  //xlib::MemoryMapped
#include "Support/Host/PrintExt.hpp"  //xlib::printArray
#include <algorithm>                  //std::iota, std::shuffle
#include <cassert>                    //assert
#include <chrono>                     //std::chrono
#include <random>                     //std::mt19937_64

namespace graph {

template<typename vid_t, typename eoff_t>
GraphStd<vid_t, eoff_t>::GraphStd(const eoff_t* csr_offsets, vid_t nV,
                                  const vid_t* csr_edges, eoff_t nE) noexcept :
                       GraphBase<vid_t, eoff_t>(nV, nE, Structure::UNDIRECTED) {
    allocate( { static_cast<size_t>(nV), static_cast<size_t>(nE),
                Structure::UNDIRECTED } );
    std::copy(csr_offsets, csr_offsets + nV + 1, _out_offsets);
    std::copy(csr_edges, csr_edges + nE, _out_edges);
    for (vid_t i = 0; i < nV; i++)
        _out_degrees[i] = csr_offsets[i + 1] - csr_offsets[i];
}

template<typename vid_t, typename eoff_t>
GraphStd<vid_t, eoff_t>::GraphStd(Structure::Enum structure) noexcept :
                       GraphBase<vid_t, eoff_t>(structure) {}

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::allocate(GInfo ginfo) noexcept {
    assert(ginfo.num_vertices > 0 && ginfo.num_edges > 0);
    if (!_structure.is_direction_set())
        _structure += ginfo.direction;
    _coo_size = ginfo.num_edges;

    size_t num_edges;
    if (_structure.is_undirected() && _store_inverse)
        num_edges = ginfo.num_edges * 2;
    else if (_structure.is_directed() && !_store_inverse &&
             ginfo.direction == Structure::UNDIRECTED) {
        _bitmask.init(_coo_size);
        _bitmask.randomize();
        num_edges = _bitmask.size();
    }
    else
        num_edges = ginfo.num_edges;

    xlib::check_overflow<vid_t>(ginfo.num_vertices);
    xlib::check_overflow<eoff_t>(num_edges);
    _nV = static_cast<vid_t>(ginfo.num_vertices);
    _nE = static_cast<eoff_t>(num_edges);

    if (_prop.is_print()) {
        const char* graph_dir = _structure.is_undirected()
                                    ? "\tGraph Structure: Undirected"
                                    : "\tGraph Structure: Directed";
        std::cout << "\nNodes: " << xlib::format(_nV) << "\tEdges: "
                  << xlib::format(_nE) << graph_dir << "\tavg. degree: "
                  << xlib::format(static_cast<double>(_nE) / _nV, 1) << "\n\n";
    }

    try {
        _out_offsets = new eoff_t[ _nV + 1 ];
        _out_edges   = new vid_t[ _nE ];
        _out_degrees = new degree_t[ _nV ]();
        _coo_edges   = new coo_t[ _coo_size ];
        if (_structure.is_undirected()) {
            _in_degrees = _out_degrees;
            _in_offsets = _out_offsets;
            _in_edges   = _out_edges;
        }
        else if (_structure.is_reverse()) {
            _in_offsets = new eoff_t[ _nV + 1 ];
            _in_edges   = new vid_t[ _nE ];
            _in_degrees = new degree_t[ _nV ]();
        }
    }
    catch (const std::bad_alloc&) {
        ERROR("OUT OF MEMORY: Graph too Large !!  V: ", _nV, " E: ", _nE)
    }
}

template<typename vid_t, typename eoff_t>
GraphStd<vid_t, eoff_t>::~GraphStd() noexcept {
    delete[] _out_offsets;
    delete[] _out_edges;
    delete[] _out_degrees;
    delete[] _coo_edges;
    if (_structure.is_directed() && _structure.is_reverse()) {
        delete[] _in_offsets;
        delete[] _in_edges;
        delete[] _in_degrees;
    }
}

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::COOtoCSR() noexcept {
    if (_bitmask.size() != 0) {
        auto tmp = new coo_t[_bitmask.size()];
        for (size_t i = 0, k = 0; i < _coo_size; i++) {
            if (_bitmask[i])
                tmp[k++] = _coo_edges[i];
        }
        delete[] _coo_edges;
        _coo_edges = tmp;
    }
    if (_prop.is_randomize()) {
        if (_prop.is_print())
            std::cout << "Randomization...\n" << std::flush;
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                    .count();
        auto random_array = new vid_t[_nV];
        std::iota(random_array, random_array + _nV, 0);
        std::shuffle(random_array, random_array + _nV, std::mt19937_64(seed));
        for (size_t i = 0; i < _coo_size; i++) {
            _coo_edges[i].first  = random_array[ _coo_edges[i].first ];
            _coo_edges[i].second = random_array[ _coo_edges[i].second ];
        }
        delete[] random_array;
    }
    if (_prop.is_sort()) {
        if (_prop.is_print())
            std::cout << "Sorting...\n" << std::flush;
        std::sort(_coo_edges, _coo_edges + _coo_size);
    }
    if (_prop.is_print())
        std::cout << "COO to CSR...\t" << std::flush;

    for (size_t i = 0; i < _coo_size; i++) {
        vid_t  src = _coo_edges[i].first;
        vid_t dest = _coo_edges[i].second;
        _out_degrees[src]++;
        if (_structure.is_undirected() && _store_inverse)
            _out_degrees[dest]++;
        else if (_structure.is_reverse())
            _in_degrees[dest]++;
    }
    _out_offsets[0] = 0;
    std::partial_sum(_out_degrees, _out_degrees + _nV, _out_offsets + 1);

    auto tmp = new degree_t[_nV]();
    for (size_t i = 0; i < _coo_size; i++) {
        vid_t  src = _coo_edges[i].first;
        vid_t dest = _coo_edges[i].second;
        _out_edges[ _out_offsets[src] + tmp[src]++ ] = dest;
        if (_structure.is_undirected() && _store_inverse)
            _out_edges[ _out_offsets[dest] + tmp[dest]++ ] = src;
    }

    if (_structure.is_directed() && _structure.is_reverse()) {
        _in_offsets[0] = 0;
        std::partial_sum(_in_degrees, _in_degrees + _nV, _in_offsets + 1);
        std::fill(tmp, tmp + _nV, 0);
        for (size_t i = 0; i < _coo_size; i++) {
            vid_t dest = _coo_edges[i].second;
            _in_edges[ _in_offsets[dest] + tmp[dest]++ ] = _coo_edges[i].first;
        }
    }
    delete[] tmp;
    if (!_structure.is_coo()) {
        delete[] _coo_edges;
        _coo_edges = nullptr;
    }
    if (_prop.is_print())
        std::cout << "Complete!\n" << std::endl;
}

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::print() const noexcept {
    for (vid_t i = 0; i < _nV; i++) {
        std::cout << "[ " << i << " ] : ";
        for (eoff_t j = _out_offsets[i]; j < _out_offsets[i + 1]; j++)
            std::cout << _out_edges[j] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::print_raw() const noexcept {
    xlib::printArray(_out_offsets, _nV + 1, "Out-Offsets  ");           //NOLINT
    xlib::printArray(_out_edges,   _nE,     "Out-Edges    ");           //NOLINT
    xlib::printArray(_out_degrees, _nV,     "Out-Degrees  ");           //NOLINT
    if (_structure.is_directed() && _structure.is_reverse()) {
        xlib::printArray(_in_offsets, _nV + 1, "In-Offsets   ");        //NOLINT
        xlib::printArray(_in_edges,   _nE,     "In-Edges     ");        //NOLINT
        xlib::printArray(_in_degrees, _nV,     "In-Degrees   ");        //NOLINT
    }
}

#if defined(__linux__)

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>
::writeBinary(const std::string& filename, bool print) const {
    size_t  base_size = sizeof(_nV) + sizeof(_nE) + sizeof(_structure);
    size_t file_size1 = (static_cast<size_t>(_nV) + 1) * sizeof(eoff_t) +
                        (static_cast<size_t>(_nE)) * sizeof(vid_t);

    bool       twice = _structure.is_directed() && _structure.is_reverse();
    size_t file_size = base_size + (twice ? file_size1 * 2 : file_size1);

    if (print) {
        std::cout << "Graph To binary file: " << filename
                << " (" << (file_size >> 20) << ") MB" << std::endl;
    }

    std::string class_id = xlib::type_name<vid_t>() + xlib::type_name<eoff_t>();
    file_size           += class_id.size();
    xlib::MemoryMapped memory_mapped(filename.c_str(), file_size,
                                     xlib::MemoryMapped::WRITE, print);

    if (_structure.is_directed() && _structure.is_reverse()) {
        memory_mapped.write(class_id.c_str(), class_id.size(),          //NOLINT
                            &_nV, 1, &_nE, 1, &_structure, 1,           //NOLINT
                            _out_offsets, _nV + 1, _in_offsets, _nV + 1,//NOLINT
                            _out_edges, _nE, _in_edges, _nE);           //NOLINT
    }
    else {
        memory_mapped.write(class_id.c_str(), class_id.size(),          //NOLINT
                            &_nV, 1, &_nE, 1, &_structure, 1,           //NOLINT
                            _out_offsets, _nV + 1, _out_edges, _nE);    //NOLINT
    }
}

#pragma clang diagnostic pop
#endif

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::writeMarket(const std::string& filename) const {
    std::ofstream fout(filename);
    fout << "%%MatrixMarket matrix coordinate pattern general"
         << "\n" << _nV << " " << _nV << " " << _nE << "\n";
    for (vid_t i = 0; i < _nV; i++) {
        for (eoff_t j = _out_offsets[i]; j < _out_offsets[i + 1]; j++)
            fout << i + 1 << " " << _out_edges[j] + 1 << "\n";
    }
    fout.close();
}

//------------------------------------------------------------------------------

template class GraphStd<int, int>;
template class GraphStd<int64_t, int64_t>;

} // namespace graph
