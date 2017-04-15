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
/**
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
#include "GraphIO/GraphStd.hpp"
#include "Support/Basic.hpp"          //ERROR
#include "Support/FileUtil.hpp"       //xlib::MemoryMapped
#include "Support/PrintExt.hpp"       //xlib::printArray
#include <algorithm>    //std::iota, std::shuffle
#include <cassert>      //assert
#include <chrono>       //std::chrono
#include <random>       //std::mt19937_64

namespace graph {

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::allocate() noexcept {
    assert(_V > 0 && _E > 0 && _structure.is_direction_set());
    try {
        _out_offsets = new off_t[ _V + 1 ];
        _out_edges   = new id_t[ _E ];
        _out_degrees = new degree_t[ _V ]();
        if (_coo_size > 0)
            _coo_edges   = new id2_t[ _coo_size ];
        if (_structure.is_undirected()) {
            _in_degrees = _out_degrees;
            _in_offsets = _out_offsets;
            _in_edges   = _out_edges;
        }
        else if (_structure.is_reverse()) {
            _in_offsets = new off_t[ _V + 1 ];
            _in_edges   = new id_t[ _E ];
            _in_degrees = new degree_t[ _V ]();
        }
    }
    catch (const std::bad_alloc&) {
        ERROR("OUT OF MEMORY: Graph too Large !!   _V: ", _V, " E: ", _E)
    }
}

template<typename id_t, typename off_t>
GraphStd<id_t, off_t>::~GraphStd() noexcept {
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

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::COOtoCSR(Property prop) noexcept {
    if (prop.is_randomize()) {
        if (prop.is_print())
            std::cout << "Randomization...\n" << std::flush;
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto random_array = new id_t[_V];
        std::iota(random_array, random_array + _V, 0);
        std::shuffle(random_array, random_array + _V, std::mt19937_64(seed));
        for (size_t i = 0; i < _coo_size; i++) {
            _coo_edges[i].first  = random_array[ _coo_edges[i].first ];
            _coo_edges[i].second = random_array[ _coo_edges[i].second ];
        }
        delete[] random_array;
    }
    if (prop.is_sort()) {
        if (prop.is_print())
            std::cout << "Sorting...\n" << std::flush;
        std::sort(_coo_edges, _coo_edges + _coo_size);
    }
    if (prop.is_print())
        std::cout << "COO to CSR...\t" << std::flush;

    for (size_t i = 0; i < _coo_size; i++) {
        id_t  src = _coo_edges[i].first;
        id_t dest = _coo_edges[i].second;
        _out_degrees[src]++;
        if (_structure.is_undirected())
            _out_degrees[dest]++;
        else if (_structure.is_reverse())
            _in_degrees[dest]++;
    }
    _out_offsets[0] = 0;
    std::partial_sum(_out_degrees, _out_degrees + _V, _out_offsets + 1);

    auto tmp = new degree_t[_V]();
    for (size_t i = 0; i < _coo_size; i++) {
        id_t  src = _coo_edges[i].first;
        id_t dest = _coo_edges[i].second;
        _out_edges[ _out_offsets[src] + tmp[src]++ ] = dest;
        if (_structure.is_undirected())
            _out_edges[ _out_offsets[dest] + tmp[dest]++ ] = src;
    }

    if (_structure.is_directed() && _structure.is_reverse()) {
        _in_offsets[0] = 0;
        std::partial_sum(_in_degrees, _in_degrees + _V, _in_offsets + 1);
        std::fill(tmp, tmp + _V, 0);
        for (size_t i = 0; i < _coo_size; i++) {
            id_t dest = _coo_edges[i].second;
            _in_edges[ _in_offsets[dest] + tmp[dest]++ ] = _coo_edges[i].first;
        }
    }
    delete[] tmp;
    if (!_structure.is_coo()) {
        delete[] _coo_edges;
        _coo_edges = nullptr;
    }
    if (prop.is_print())
        std::cout << "Complete!\n" << std::endl;
}

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::print() const noexcept {
    for (id_t i = 0; i < _V; i++) {
        std::cout << "[ " << i << " ] : ";
        for (off_t j = _out_offsets[i]; j < _out_offsets[i + 1]; j++)
            std::cout << _out_edges[j] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::print_raw() const noexcept {
    xlib::printArray(_out_offsets, _V + 1, "Out-Offsets  ");            //NOLINT
    xlib::printArray(_out_edges,   _E,     "Out-Edges    ");            //NOLINT
    xlib::printArray(_out_degrees, _V,     "Out-Degrees  ");            //NOLINT
    if (_structure.is_directed() && _structure.is_reverse()) {
        xlib::printArray(_in_offsets, _V + 1, "In-Offsets   ");         //NOLINT
        xlib::printArray(_in_edges,   _E,     "In-Edges     ");         //NOLINT
        xlib::printArray(_in_degrees, _V,     "In-Degrees   ");         //NOLINT
    }
}

#if defined(__linux__)


template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::toBinary(const std::string& filename, bool print)
                                     const {
    size_t  base_size = sizeof(_V) + sizeof(_E) + sizeof(_structure);
    size_t file_size1 = (static_cast<size_t>(_V) + 1) * sizeof(off_t) +
                        (static_cast<size_t>(_E)) * sizeof(id_t);

    bool       twice = _structure.is_directed() && _structure.is_reverse();
    size_t file_size = base_size + (twice ? file_size1 * 2 : file_size1);

    if (print) {
        std::cout << "Graph To binary file: " << filename
                << " (" << (file_size >> 20) << ") MB" << std::endl;
    }

    std::string class_id = xlib::type_name<id_t>() + xlib::type_name<off_t>();
    file_size           += class_id.size();
    xlib::MemoryMapped memory_mapped(filename.c_str(), file_size,
                                     xlib::MemoryMapped::WRITE, print);

    if (_structure.is_directed() && _structure.is_reverse()) {
        memory_mapped.write(class_id.c_str(), class_id.size(),
                            &_V, 1, &_E, 1, &_structure, 1,
                            _out_offsets, _V + 1, _in_offsets, _V + 1,
                            _out_edges, _E, _in_edges, _E);
    }
    else {
        memory_mapped.write(class_id.c_str(), class_id.size(),
                            &_V, 1, &_E, 1, &_structure, 1,
                            _out_offsets, _V + 1, _out_edges, _E);
    }
}

#pragma clang diagnostic pop
#endif

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::toMarket(const std::string& filename, bool)
                                     const {
    std::ofstream fout(filename);
    fout << "%%MatrixMarket matrix coordinate pattern general"
         << "\n" << _V << " " << _V << " " << _E << "\n";
    for (id_t i = 0; i < _V; i++) {
        for (off_t j = _out_offsets[i]; j < _out_offsets[i + 1]; j++)
            fout << i + 1 << " " << _out_edges[j] + 1 << "\n";
    }
    fout.close();
}

//------------------------------------------------------------------------------

template class GraphStd<int, int>;
template class GraphStd<int64_t, int64_t>;

} // namespace graph
