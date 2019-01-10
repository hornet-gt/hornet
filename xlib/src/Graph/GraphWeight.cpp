/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
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
#include "Graph/GraphWeight.hpp"
#include "Host/Basic.hpp"     //ERROR
#include "Host/FileUtil.hpp"  //xlib::MemoryMapped
#include "Host/PrintExt.hpp"  //xlib::printArray
#include <algorithm>          //std::iota, std::shuffle
#include <cassert>            //assert
#include <chrono>             //std::chrono
#include <random>             //std::mt19937_64

namespace graph {

template<typename vid_t, typename eoff_t, typename weight_t>
GraphWeight<vid_t, eoff_t, weight_t>
::GraphWeight(const eoff_t* csr_offsets, vid_t nV,
              const vid_t* csr_edges, eoff_t nE,
              const weight_t* csr_weights) noexcept :
                  GraphStd<vid_t, eoff_t>(csr_offsets, nV, csr_edges, nE) {
    _out_weights = new weight_t[ _nE ];
    _in_weights = _out_weights;
    std::copy(csr_weights, csr_weights + nE, _out_weights);
}

template<typename vid_t, typename eoff_t, typename weight_t>
GraphWeight<vid_t, eoff_t, weight_t>
::GraphWeight(StructureProp structure) noexcept :
                    GraphStd<vid_t, eoff_t>(std::move(structure)) {}

template<typename vid_t, typename eoff_t, typename weight_t>
GraphWeight<vid_t, eoff_t, weight_t>
::GraphWeight(const char* filename, const ParsingProp& property) noexcept :
                    GraphStd<vid_t, eoff_t>(filename, property) {}

template<typename vid_t, typename eoff_t, typename weight_t>
GraphWeight<vid_t, eoff_t, weight_t>
::GraphWeight(StructureProp structure, const char* filename,
              const ParsingProp& property) noexcept :
                    GraphStd<vid_t, eoff_t>(structure, filename, property) {}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::allocate(const GInfo& ginfo) noexcept {
    GraphStd<vid_t, eoff_t>::allocateAux(ginfo);
    _coo_size = _nE;
    try {
        _coo_edges = new coo_t[ _nE ];
        _out_weights = new weight_t[ _nE ];
        if (_structure.is_undirected()) {
            _in_weights = _out_weights;
        }
        else if (_structure.is_reverse()) {
            _in_weights = new weight_t[ _nE ];
        }
    }
    catch (const std::bad_alloc&) {
        ERROR("OUT OF MEMORY: Graph too Large !!   _nV: ", _nV, " E: ", _nE)
    }
}

template<typename vid_t, typename eoff_t, typename weight_t>
GraphWeight<vid_t, eoff_t, weight_t>::~GraphWeight() noexcept {
    delete[] _out_weights;
    if (_structure.is_directed() && _structure.is_reverse())
        delete[] _in_weights;
}

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>::COOtoCSR() noexcept {
    if (_directed_to_undirected || _stored_undirected) {
        eoff_t half = _nE / 2;
        auto      k = half;
        for (eoff_t i = 0; i < half; i++) {
            auto src = std::get<0>(_coo_edges[i]);
            auto dst = std::get<1>(_coo_edges[i]);
            if (src == dst)
                continue;
            _coo_edges[k++] = coo_t(dst, src, std::get<2>(_coo_edges[i]));
        }
        if (_prop.is_print() && _nE != k) {
            std::cout << "Double self-loops removed.  E: " << xlib::format(k)
                      << "\n";
        }
        _nE = k;
    }

    if (_directed_to_undirected) {
        if (_prop.is_print()) {
            if (_directed_to_undirected)
                std::cout << "Directed to Undirected: ";
            std::cout << "Removing duplicated edges..." << std::flush;
        }
        std::sort(_coo_edges, _coo_edges + _nE);
        auto   last = std::unique(_coo_edges, _coo_edges + _nE);
        auto new_nE = std::distance(_coo_edges, last);
        if (_prop.is_print() && new_nE != _nE) {
            std::cout << "(" << xlib::format(_nE - new_nE) << " edges removed)"
                      << std::endl;
        }
        _nE = new_nE;
    }
    else if (_undirected_to_directed) {
        std::cout << "Undirected to Directed: Removing random edges..."
                  << std::endl;
        for (eoff_t i = 0, k = 0; i < _nE; i++) {
            if (_bitmask[i])
                _coo_edges[k++] = _coo_edges[i];
        }
        _bitmask.free();
    }

    if (_prop.is_randomize()) {
        if (_prop.is_print())
            std::cout << "Randomization..." << std::endl;
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                    .count();
        auto random_array = new vid_t[_nV];
        std::iota(random_array, random_array + _nV, 0);
        std::shuffle(random_array, random_array + _nV, std::mt19937_64(seed));
        for (eoff_t i = 0; i < _nE; i++) {
            vid_t  src = std::get<0>(_coo_edges[i]);
            vid_t dest = std::get<1>(_coo_edges[i]);
            std::get<0>(_coo_edges[i]) = random_array[ src ];
            std::get<1>(_coo_edges[i]) = random_array[ dest ];
        }
        delete[] random_array;
    }
    if (_prop.is_sort() && (!_directed_to_undirected || _prop.is_randomize())) {
        if (_prop.is_print())
            std::cout << "Sorting..." << std::endl;
        std::sort(_coo_edges, _coo_edges + _nE);
    }
    //--------------------------------------------------------------------------
    if (_prop.is_print())
        std::cout << "COO to CSR...\t" << std::flush;

    if (_structure.is_reverse() && _structure.is_directed()) {
        for (eoff_t i = 0; i < _nE; i++) {
            _out_degrees[std::get<0>(_coo_edges[i])]++;
            _in_degrees[std::get<1>(_coo_edges[i])]++;
        }
    }
    else {
        for (eoff_t i = 0; i < _nE; i++)
            _out_degrees[std::get<0>(_coo_edges[i])]++;
    }

    _out_offsets[0] = 0;
    std::partial_sum(_out_degrees, _out_degrees + _nV, _out_offsets + 1);

    auto tmp = new degree_t[_nV]();
    for (size_t i = 0; i < _coo_size; i++) {
        vid_t    src = std::get<0>(_coo_edges[i]);
        vid_t   dest = std::get<1>(_coo_edges[i]);
        auto offset1 = _out_offsets[src] + tmp[src]++;
        _out_edges[ offset1 ]   = dest;
        _out_weights[ offset1 ] = std::get<2>(_coo_edges[i]);
    }

    if (_structure.is_directed() && _structure.is_reverse()) {
        _in_offsets[0] = 0;
        std::partial_sum(_in_degrees, _in_degrees + _nV, _in_offsets + 1);
        std::fill(tmp, tmp + _nV, 0);
        for (size_t i = 0; i < _coo_size; i++) {
            auto    src = std::get<0>(_coo_edges[i]);
            auto    dst = std::get<1>(_coo_edges[i]);
            auto offset = _in_offsets[dst] + tmp[dst]++;
            _in_edges[ offset ]   = src;
            _in_weights[ offset ] = std::get<2>(_coo_edges[i]);
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

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>::print() const noexcept {
    for (vid_t i = 0; i < _nV; i++) {
        std::cout << "[ " << i << " ] : ";
        for (eoff_t j = _out_offsets[i]; j < _out_offsets[i + 1]; j++) {
            std::cout << "(" << _out_edges[j] << ","
                      << _out_weights[j] << ")  ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>::print_raw() const noexcept {
    xlib::printArray(_out_offsets, _nV + 1, "Out-Offsets  ");           //NOLINT
    xlib::printArray(_out_edges,   _nE,     "Out-Edges    ");           //NOLINT
    xlib::printArray(_out_weights, _nE,     "Out-Weights  ");           //NOLINT
    xlib::printArray(_out_degrees, _nV,     "Out-Degrees  ");           //NOLINT
    if (_structure.is_directed() && _structure.is_reverse()) {
        xlib::printArray(_in_offsets, _nV + 1, "In-Offsets   ");        //NOLINT
        xlib::printArray(_in_edges,   _nE,     "In-Edges     ");        //NOLINT
        xlib::printArray(_in_weights, _nE,     "In-Weights  ");         //NOLINT
        xlib::printArray(_in_degrees, _nV,     "In-Degrees   ");        //NOLINT
    }
}

#if defined(__linux__)

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::toBinary(const std::string& filename, bool print) const {
    size_t  base_size = sizeof(_nV) + sizeof(_nE) + sizeof(_structure);
    size_t file_size1 = (static_cast<size_t>(_nV) + 1) * sizeof(eoff_t) +
                        (static_cast<size_t>(_nE)) * sizeof(vid_t) +
                        (static_cast<size_t>(_nE)) * sizeof(weight_t);

    bool       twice = _structure.is_directed() && _structure.is_reverse();
    size_t file_size = base_size + (twice ? file_size1 * 2 : file_size1);

    if (print) {
        std::cout << "Graph To binary file: " << filename
                << " (" << (file_size >> 20) << ") MB" << std::endl;
    }

    std::string class_id = xlib::type_name<vid_t>() + xlib::type_name<eoff_t>() +
                           xlib::type_name<weight_t>();
    file_size           += class_id.size();
    xlib::MemoryMapped memory_mapped(filename.c_str(), file_size,
                                     xlib::MemoryMapped::WRITE, print);

    if (_structure.is_directed() && _structure.is_reverse()) {
        memory_mapped.write(class_id.c_str(), class_id.size(),          //NOLINT
                            &_nV, 1, &_nE, 1, &_structure, 1,           //NOLINT
                            _out_offsets, _nV + 1, _in_offsets, _nV + 1,//NOLINT
                            _out_edges, _nE, _in_edges, _nE,            //NOLINT
                            _out_weights, _nE, _in_weights, _nE);       //NOLINT
    }
    else {
        memory_mapped.write(class_id.c_str(), class_id.size(),          //NOLINT
                            &_nV, 1, &_nE, 1, &_structure, 1,           //NOLINT
                            _out_offsets, _nV + 1, _out_edges, _nE,     //NOLINT
                            _out_weights, _nE);
    }
}

#endif

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::toMarket(const std::string& filename) const {
    std::ofstream fout(filename);
    fout << "%%MatrixMarket matrix coordinate pattern general"
         << "\n" << _nV << " " << _nV << " " << _nE << "\n";
    for (vid_t i = 0; i < _nV; i++) {
        for (eoff_t j = _out_offsets[i]; j < _out_offsets[i + 1]; j++) {
            fout << i + 1 << " " << _out_edges[j] + 1 << " "
                 << _out_weights[j] << "\n";
        }
    }
    fout.close();
}

//------------------------------------------------------------------------------

template class GraphWeight<int, int, int>;
template class GraphWeight<int, int, float>;
template class GraphWeight<int64_t, int64_t, int64_t>;

} // namespace graph
