/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date June, 2017
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
#include "Graph/GraphStd.hpp"
#include "Host/Basic.hpp"      //ERROR
#include "Host/FileUtil.hpp"   //xlib::MemoryMapped
#include "Host/Numeric.hpp"    //xlib::per_cent
#include "Host/PrintExt.hpp"   //xlib::printArray
#include "Host/Statistics.hpp" //xlib::average
#include <algorithm>           //std::iota, std::shuffle
#include <cassert>             //assert
#include <chrono>              //std::chrono
#include <random>              //std::mt19937_64

namespace graph {

template<typename vid_t, typename eoff_t>
GraphStd<vid_t, eoff_t>::GraphStd(const eoff_t* csr_offsets, vid_t nV,
                                  const vid_t* csr_edges, eoff_t nE) noexcept :
                  GraphBase<vid_t, eoff_t>(nV, nE, structure_prop::UNDIRECTED) {
    allocate( { static_cast<size_t>(nV), static_cast<size_t>(nE),
                static_cast<size_t>(nE), structure_prop::UNDIRECTED } );
    std::copy(csr_offsets, csr_offsets + nV + 1, _out_offsets);
    std::copy(csr_edges, csr_edges + nE, _out_edges);
    for (vid_t i = 0; i < nV; i++)
        _out_degrees[i] = csr_offsets[i + 1] - csr_offsets[i];
}

template<typename vid_t, typename eoff_t>
GraphStd<vid_t, eoff_t>::GraphStd(StructureProp structure) noexcept :
                       GraphBase<vid_t, eoff_t>(std::move(structure)) {}

template<typename vid_t, typename eoff_t>
GraphStd<vid_t, eoff_t>::GraphStd(const char* filename,
                                  const ParsingProp& property)
                                  noexcept : GraphBase<vid_t, eoff_t>() {
    GraphBase<vid_t, eoff_t>::read(filename, property);
}

template<typename vid_t, typename eoff_t>
GraphStd<vid_t, eoff_t>::GraphStd(StructureProp structure,
                                  const char* filename,
                                  const ParsingProp& property)
                                  noexcept :
                                      GraphBase<vid_t, eoff_t>(structure) {
    GraphBase<vid_t, eoff_t>::read(filename, property);
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::allocate(const GInfo& ginfo) noexcept {
    allocateAux(ginfo);
    try {
        size_t allocate_size = _undirected_to_directed ? ginfo.num_edges : _nE;
        _coo_edges = new coo_t[ allocate_size ];
    } catch (const std::bad_alloc&) {
        ERROR("OUT OF MEMORY: Graph too Large !!  V: ", _nV, " E: ", _nE)
    }
}

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::allocateAux(const GInfo& ginfo) noexcept {
    assert(ginfo.num_vertices > 0 && ginfo.num_edges > 0);
    if (!_structure.is_direction_set())
        _structure += ginfo.direction;
    _undirected_to_directed = ginfo.direction == structure_prop::UNDIRECTED &&
                              _structure.is_directed();
    _directed_to_undirected = ginfo.direction == structure_prop::DIRECTED &&
                              _structure.is_undirected();
    size_t new_num_edges = ginfo.num_edges;
    if (_directed_to_undirected)
        new_num_edges = ginfo.num_edges * 2;
    else if (_undirected_to_directed) {
        _bitmask.init(ginfo.num_edges);
        _bitmask.randomize(_seed);
        new_num_edges = _bitmask.size();
    }

    xlib::check_overflow<vid_t>(ginfo.num_vertices);
    xlib::check_overflow<eoff_t>(new_num_edges);
    _nV = static_cast<vid_t>(ginfo.num_vertices);
    _nE = static_cast<eoff_t>(new_num_edges);

    if (_prop.is_print()) {
        const char* const dir[] = { "Structure: Undirected   ",
                                    "Structure: Directed     " };
        const char* graph_dir = ginfo.direction == structure_prop::UNDIRECTED
                                    ? dir[0] : dir[1];
        auto avg = static_cast<double>(ginfo.num_edges) / _nV;
        std::cout << "\n@File    V: " << std::left << std::setw(14)
                  << xlib::format(_nV)  << "E: " << std::setw(14)
                  << xlib::format(ginfo.num_edges) << graph_dir
                  << "avg. deg: " << xlib::format(avg, 1);
        if (_directed_to_undirected || _undirected_to_directed) {
            graph_dir =  _structure.is_undirected() ? dir[0] : dir[1];
            avg = static_cast<double>(new_num_edges) / _nV;
            std::cout << "\n@User    V: "  << std::left << std::setw(14)
                      << xlib::format(_nV) << "E: " << std::setw(14)
                      << xlib::format(new_num_edges) << graph_dir
                      << "avg. deg: " << xlib::format(avg) << "\n";
        }
        else
            assert(new_num_edges == ginfo.num_edges);
        std::cout << std::right << std::endl;
    }

    try {
        _out_offsets = new eoff_t[ _nV + 1 ];
        _out_edges   = new vid_t[ _nE ];
        _out_degrees = new degree_t[ _nV ]();
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
    if (_directed_to_undirected || _stored_undirected) {
        eoff_t half = _nE / 2;
        auto      k = half;
        for (eoff_t i = 0; i < half; i++) {
            auto src = _coo_edges[i].first;
            auto dst = _coo_edges[i].second;
            if (src == dst)
                continue;
            _coo_edges[k++] = {dst, src};
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
            _coo_edges[i].first  = random_array[ _coo_edges[i].first ];
            _coo_edges[i].second = random_array[ _coo_edges[i].second ];
        }
        delete[] random_array;
    }

    //--------------------------------------------------------------------------
    if (_prop.is_print())
        std::cout << "COO to CSR...\t" << std::flush;

    if (_structure.is_directed() && _structure.is_reverse()) {
        for (eoff_t i = 0; i < _nE; i++) {
            _out_degrees[_coo_edges[i].first]++;
            _in_degrees[_coo_edges[i].second]++;
        }
    }
    else {
        for (eoff_t i = 0; i < _nE; i++)
            _out_degrees[_coo_edges[i].first]++;
    }

    if (_prop.is_directed_by_degree()) {
        if (_prop.is_print())
            std::cout << "Creating degree-directed graph ..." << std::endl;
        auto coo_edges_old = _coo_edges;
        auto in_degrees_old = _in_degrees;
        auto out_degrees_old = _out_degrees;
        auto in_degrees_old_inaccessible = false;
        eoff_t counter = 0;
        vid_t u, v;
        degree_t deg_u, deg_v;
        coo_t* coo_edges_tmp = new coo_t[_nE];
        degree_t* _out_degrees_tmp = new degree_t[_nV]();
        degree_t*  _in_degrees_tmp;
        for (eoff_t i=0; i<_nE; i++) {
            u = _coo_edges[i].first;
            v = _coo_edges[i].second;
            deg_u = _out_degrees[u];
            deg_v = _out_degrees[v];
            if ((deg_u < deg_v) || ((deg_u == deg_v) && (u < v))) {
                coo_edges_tmp[counter++] = {u, v}; 
            }
        }
        _coo_edges = coo_edges_tmp;
        _nE = counter;
        
        if (_structure.is_reverse()) {
            _in_degrees_tmp = new degree_t[_nV]();
            if (_structure.is_directed()) {
                for (eoff_t i = 0; i < _nE; i++) {
                    _out_degrees_tmp[_coo_edges[i].first]++;
                    _in_degrees_tmp[_coo_edges[i].second]++;
                }
            }
            else {
                for (eoff_t i = 0; i < _nE; i++)
                    _out_degrees_tmp[_coo_edges[i].first]++;
            }
            _in_degrees = _in_degrees_tmp;
            in_degrees_old_inaccessible = true;
        }
        else {
            for (eoff_t i = 0; i < _nE; i++)
                _out_degrees_tmp[_coo_edges[i].first]++;
        }
        _out_degrees = _out_degrees_tmp;

        delete[] coo_edges_old;

        if (in_degrees_old == out_degrees_old) {/* this class is poorly designed, it is very non-intuitive to consider that _in_degrees and _out_degrees can point to a same memory block, if only one is necessary, another should be set to nullptr. */
            delete[] out_degrees_old;
        }
        else {
            delete[] out_degrees_old;
            if (in_degrees_old_inaccessible == true) {
                delete[] in_degrees_old;
            }
        }
    }

    if (_prop.is_sort() && (!_directed_to_undirected || _prop.is_randomize())) {
        if (_prop.is_print())
            std::cout << "Sorting..." << std::endl;
        std::sort(_coo_edges, _coo_edges + _nE);
    }

    if (_prop.is_rm_singleton() && _structure.is_reverse()) {
        if (_prop.is_print())
            std::cout << "\nRelabeling...\t" << std::flush;
        vid_t k = 0;
        auto labels = new vid_t[_nV];
        for (vid_t i = 0; i < _nV; i++) {
            if (_out_degrees[i] != 0 && _in_degrees[i] != 0) {
                labels[i] = k;
                _out_degrees[k] = _out_degrees[i];
                _in_degrees[k]  = _in_degrees[i];
                k++;
            }
        }
        for (eoff_t i = 0; i < _nE; i++) {
            _coo_edges[i].first  = labels[_coo_edges[i].first];
            _coo_edges[i].second = labels[_coo_edges[i].second];
        }
        delete[] labels;
        _nV = k;
    }

    _out_offsets[0] = 0;
    std::partial_sum(_out_degrees, _out_degrees + _nV, _out_offsets + 1);

    auto tmp = new degree_t[_nV]();
    for (eoff_t i = 0; i < _nE; i++) {
        vid_t  src = _coo_edges[i].first;
        vid_t dest = _coo_edges[i].second;
        _out_edges[ _out_offsets[src] + tmp[src]++ ] = dest;
    }

    if (_structure.is_directed() && _structure.is_reverse()) {
        _in_offsets[0] = 0;
        std::partial_sum(_in_degrees, _in_degrees + _nV, _in_offsets + 1);
        std::fill(tmp, tmp + _nV, 0);
        for (eoff_t i = 0; i < _nE; i++) {
            auto dst = _coo_edges[i].second;
            _in_edges[ _in_offsets[dst] + tmp[dst]++ ] = _coo_edges[i].first;
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
    using namespace structure_prop;
    size_t  base_size = sizeof(_nV) + sizeof(_nE) + sizeof(_structure);
    size_t file_size1 = (static_cast<size_t>(_nV) + 1) * sizeof(eoff_t) +
                        (static_cast<size_t>(_nE)) * sizeof(vid_t);

    bool       twice = _structure.is_directed() && _structure.is_reverse();
    size_t file_size = base_size + (twice ? file_size1 * 2 : file_size1);

    if (print) {
        std::cout << "Graph to binary file: " << filename
                << " (" << (file_size >> 20) << ") MB" << std::endl;
    }

    std::string class_id = xlib::type_name<vid_t>() + xlib::type_name<eoff_t>();
    file_size           += class_id.size();
    xlib::MemoryMapped memory_mapped(filename.c_str(), file_size,
                                     xlib::MemoryMapped::WRITE, print);

    if (_structure.is_directed() && _structure.is_reverse()) {
        auto struct_tmp = DIRECTED | ENABLE_INGOING;
        memory_mapped.write(class_id.c_str(), class_id.size(),          //NOLINT
                            &_nV, 1, &_nE, 1, &struct_tmp, 1,           //NOLINT
                            _out_offsets, _nV + 1, _in_offsets, _nV + 1,//NOLINT
                            _out_edges, _nE, _in_edges, _nE);           //NOLINT
    }
    else {
        auto struct_tmp = DIRECTED;
        memory_mapped.write(class_id.c_str(), class_id.size(),          //NOLINT
                            &_nV, 1, &_nE, 1, &struct_tmp, 1,           //NOLINT
                            _out_offsets, _nV + 1, _out_edges, _nE);    //NOLINT
    }
}

#endif

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::writeMarket(const std::string& filename,
                                          bool print) const {
    if (print)
        std::cout << "Graph to Market format file: " << filename << std::endl;
    std::ofstream fout(filename);
    fout << "%%MatrixMarket matrix coordinate pattern general\n"
         << _nV << " " << _nV << " " << _nE << "\n";
    for (vid_t i = 0; i < _nV; i++) {
        for (auto j = _out_offsets[i]; j < _out_offsets[i + 1]; j++)
            fout << i + 1 << " " << _out_edges[j] + 1 << "\n";
    }
    fout.close();
}

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::writeDimacs10th(const std::string& filename,
                                              bool print) const {
    if (print) {
        std::cout << "Graph to Dimacs10th format file: " << filename
                  << std::endl;
    }
    std::ofstream fout(filename);
    fout << _nV << " " << _nE << " 100\n";
    for (vid_t i = 0; i < _nV; i++) {
        for (auto j = _out_offsets[i]; j < _out_offsets[i + 1]; j++) {
            fout << _out_edges[j] + 1;
            if (j < _out_offsets[i + 1] - 1)
                fout << " ";
        }
        fout << "\n";
    }
    fout.close();
}
//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::print_degree_distrib() const noexcept {
    const int MAX_LOG = 32;
    int distribution[MAX_LOG] = {};
    int   cumulative[MAX_LOG] = {};
    int      percent[MAX_LOG];
    int cumulative_percent[MAX_LOG];
    for (auto i = 0; i < _nV; i++) {
        auto degree = _out_degrees[i];
        if (degree == 0) continue;
        auto log_value = xlib::log2(degree);
        distribution[log_value]++;
        cumulative[log_value] += degree;
    }
    for (auto i = 0; i < MAX_LOG; i++) {
        percent[i] = xlib::per_cent(distribution[i], _nV);
        cumulative_percent[i] = xlib::per_cent(cumulative[i], _nE);
    }
    int pos = MAX_LOG;
    while (pos >= 0 && distribution[--pos] == 0);

    xlib::IosFlagSaver tmp1;
    xlib::ThousandSep  tmp2;
    using namespace std::string_literals;

    std::cout << "Degree distribution:" << std::setprecision(1) << "\n\n";
    for (auto i = 0; i <= pos; i++) {
        std::string exp = "  (2^"s + std::to_string(i) + ")"s;
        std::cout << std::right << std::setw(9)  << (1 << i)
                  << std::left  << std::setw(8)  << exp
                  << std::right << std::setw(12) << distribution[i]
                  << std::right << std::setw(5)  << percent[i] << " %\n";
    }
    std::cout << "\nEdge distribution:" << std::setprecision(1) << "\n\n";
    for (auto i = 0; i <= pos; i++) {
        std::string exp = "  (2^"s + std::to_string(i) + ")"s;
        std::cout << std::right << std::setw(9)  << (1 << i)
                  << std::left  << std::setw(8)  << exp
                  << std::right << std::setw(12) << cumulative[i]
                  << std::right << std::setw(5)  << cumulative_percent[i]
                  << " %\n";
    }
    std::cout << std::endl;
}

template<typename vid_t, typename eoff_t>
typename GraphStd<vid_t, eoff_t>::GraphAnalysisProp
GraphStd<vid_t, eoff_t>::_collect_analysis() const noexcept {
    GraphAnalysisProp prop;
    prop.std_dev = xlib::std_deviation(_out_degrees, _out_degrees + _nV);
    prop.gini    = xlib::gini_coefficient(_out_degrees, _out_degrees + _nV);

    xlib::Bitmask rings(_nV);
    for (auto i = 0; i < _nV; i++) {
        for (auto j = _out_offsets[i]; j < _out_offsets[i + 1]; j++) {
            if (_out_edges[j] == i) {
                rings[i] = true;
                break;
            }
        }
    }
    prop.num_rings  = rings.size();
    bool is_inverse = _in_degrees != nullptr;
    degree_t count  = 0;

    for (vid_t i = 0; i < _nV; i++) {
        if (_out_degrees[i] > prop.max_out_degree)
            prop.max_out_degree = _out_degrees[i];
        if (_out_degrees[i] == 0) {
            prop.out_degree_0++;
            count++;
            if (count > prop.max_consec_0)
                prop.max_consec_0 = count;
        }
        else
            count = 0;

        if (_out_degrees[i] == 1)
            prop.out_degree_1++;
        if (((_out_degrees[i] == 2 && is_undirected()) ||
                (_out_degrees[i] == 1 && is_directed())) && rings[i]) {
            prop.out_leaf++;
        }
        if (is_undirected() && (_out_degrees[i] == 0 ||
                (_out_degrees[i] == 1 && rings[i]))) {
            prop.singleton++;
        }
        if (!is_inverse)
            continue;
        if (_in_degrees[i] > prop.max_in_degree)
            prop.max_in_degree = _in_degrees[i];
        if (_in_degrees[i] == 0)
            prop.in_degree_0++;
        if (_in_degrees[i] == 1)
            prop.in_degree_1++;
        if (((_in_degrees[i] == 2 && is_undirected()) ||
                (_in_degrees[i] == 1 && is_directed())) && rings[i]) {
            prop.in_leaf++;
        }
        if (is_undirected())
            continue;
        if ((_out_degrees[i] == 0 && _in_degrees[i] == 0) ||
                (_out_degrees[i] == 1 && _in_degrees[i] == 1 && rings[i])) {
            prop.singleton++;
        }
    }
    return prop;
}

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>
::write_analysis(const char* filename) const noexcept {
    auto prop = _collect_analysis();
    std::ofstream fout(filename, std::ofstream::out | std::ofstream::app);

    auto dir = is_directed() ? "D" : "U";
    fout << _graph_name  << "\t" << dir << "\t"
         << _nV << "\t" << _nE << "\t"
         << prop.std_dev        << "\t" << prop.gini          << "\t"
         << prop.max_out_degree << "\t" << prop.max_in_degree << "\t"
         << prop.num_rings      << "\t" << prop.singleton     << "\t"
         << prop.out_degree_0   << "\t" << prop.in_degree_0   << "\t"
         << prop.out_degree_1   << "\t" << prop.in_degree_1   << "\t"
         << prop.max_consec_0   << "\t"
         << prop.out_leaf       << "\t" << prop.in_leaf << "\n";
    fout.close();
}

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::print_analysis() const noexcept {
    auto prop     = _collect_analysis();
    auto avg      = static_cast<float>(_nE) / static_cast<float>(_nV);
    auto sparsity = 1.0f - static_cast<float>(_nE) / static_cast<float>(
                     static_cast<uint64_t>(_nV) * static_cast<uint64_t>(_nV));
    auto variance_coeff = prop.std_dev / avg;
    auto ring_percent   = xlib::per_cent(prop.num_rings, _nV);
    bool is_inverse     = _in_degrees != nullptr;

    auto out_degree_0_percent = xlib::per_cent(prop.out_degree_0, _nV);
    auto out_degree_1_percent = xlib::per_cent(prop.out_degree_1, _nV);
    auto in_degree_0_percent  = xlib::per_cent(prop.in_degree_0, _nV);
    auto in_degree_1_percent  = xlib::per_cent(prop.in_degree_1, _nV);
    auto singleton_percent    = xlib::per_cent(prop.singleton, _nV);
    auto out_leaf_percent     = xlib::per_cent(prop.out_leaf, _nV);
    auto in_leaf_percent      = xlib::per_cent(prop.in_leaf, _nV);

    const int W2 = 5;
    const int W3 = 8;
    xlib::IosFlagSaver tmp1;
    xlib::ThousandSep tmp2;

    std::cout << std::setprecision(1) << std::fixed
              << "Degree analysis:\n"
              << "\n Average:              " << std::setw(W2) << avg
              << "\n Std. Deviation:       " << std::setw(W2) << prop.std_dev
              << "\n Coeff. of variation:  " << std::setw(W2) << variance_coeff
              << "\n Gini Coeff:           " << std::setprecision(2)
                                             << std::setw(W2) << prop.gini
                                             << std::setprecision(1)
              << "\n Sparsity:             " << std::setw(W2)
                                             << sparsity * 100.0f << " %\n"
              << "\n Max Out-Degree:       " << std::setw(W2)
                                             << prop.max_out_degree;
    if (is_directed() && is_inverse)
        std::cout << "\n Max In-Degree:        " << std::setw(W2)
                                                 << prop.max_in_degree;

    std::cout << "\n Rings:                " << std::setw(W2) << prop.num_rings
                                             << std::setw(W3)
                                             << ring_percent << " %"
              << "\n Out-Degree = 0:       " << std::setw(W2)
                                             << prop.out_degree_0
                                             << std::setw(W3)
                                             << out_degree_0_percent << " %";
    if (is_directed() && is_inverse) {
        std::cout << "\n In-Degree = 0:        " << std::setw(W2)
                                                 << prop.in_degree_0
                                                 << std::setw(W3)
                                                 << in_degree_0_percent << " %";
    }
    std::cout << "\n Out-Degree = 1:       " << std::setw(W2)
                                             << prop.out_degree_1
                                             << std::setw(W3)
                                             << out_degree_1_percent << " %";
    if (is_directed() && is_inverse) {
        std::cout << "\n In-Degree = 1:        " << std::setw(W2)
                                                 << prop.in_degree_1
                                                 << std::setw(W3)
                                                 << in_degree_1_percent
                                                 << " %\n";
    }
    std::cout << "\n Max. Consec. 0:       " << std::setw(W2)
                                             << prop.max_consec_0;

    if (is_inverse) {
        std::cout << "\n Singleton:            " << std::setw(W2)
                                                 << prop.singleton
                                                 << std::setw(W3)
                                                 << singleton_percent << " %";
    }
    std::cout << "\n Out-Leaf:             " << std::setw(W2)
                                             << prop.out_leaf << std::setw(W3)
                                             << out_leaf_percent << " %";
    if (is_directed() && is_inverse) {
        std::cout << "\n In-Leaf:              " << std::setw(W2)
                                                 << prop.in_leaf
                                                 << std::setw(W3)
                                                 << in_leaf_percent << " %\n";
    }
    std::cout << std::endl;
}

//------------------------------------------------------------------------------

template class GraphStd<int16_t, int16_t>;
template class GraphStd<int, int>;
template class GraphStd<int64_t, int64_t>;

} // namespace graph
