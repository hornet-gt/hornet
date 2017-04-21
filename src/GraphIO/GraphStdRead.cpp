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
#include "Support/Algorithm.hpp"  //xlib::UniqueMap
#include "Support/FileUtil.hpp"   //xlib::skip_lines, xlib::Progress
#include <cstring>                //std::strtok
#include <sstream>                //std::istringstream
#include <vector>                 //std::vector

namespace graph {

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::readMarket(std::ifstream& fin, Property prop) {
    _coo_size = GraphBase<id_t, off_t>::getMarketHeader(fin);
    allocate();
    xlib::Progress progress(_coo_size);

    for (size_t lines = 0; lines < _coo_size; lines++) {
        id_t index1, index2;
        fin >> index1 >> index2;
        _coo_edges[lines] = std::make_pair(index1 - 1, index2 - 1);

        if (prop.is_print())
            progress.next(lines);
        xlib::skip_lines(fin);
    }
    COOtoCSR(prop);
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::readDimacs9(std::ifstream& fin, Property prop) {
    _coo_size = GraphBase<id_t, off_t>::getDimacs9Header(fin);
    allocate();
    xlib::Progress progress(_coo_size);

    int c;
    size_t lines = 0;
    while ((c = fin.peek()) != std::char_traits<char>::eof()) {
        if (c == static_cast<int>('a')) {
            id_t index1, index2;
            xlib::skip_words(fin);
            fin >> index1 >> index2;

            _coo_edges[lines] = std::make_pair(index1 - 1, index2 - 1);
            if (prop.is_print())
                progress.next(lines);
            lines++;
        }
        xlib::skip_lines(fin);
    }
    COOtoCSR(prop);
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::readKonect(std::ifstream& fin, Property prop) {
    GraphBase<id_t, off_t>::getKonectHeader(fin);
    xlib::UniqueMap<id_t> unique_map;
    std::vector<id2_t> coo_edges_vect(32768);

    size_t n_of_lines = 0;
    while (fin.good()) {
        id_t index1, index2;
        fin >> index1 >> index2;
        unique_map.insertValue(index1);
        unique_map.insertValue(index2);

        coo_edges_vect.push_back(std::make_pair(index1 - 1, index2 - 1));
        n_of_lines++;
    }
    _V = static_cast<id_t>(unique_map.size());
    _E = static_cast<off_t>(n_of_lines);
    _coo_size = n_of_lines;
    allocate();
    std::copy(coo_edges_vect.begin(), coo_edges_vect.end(), _coo_edges);
    COOtoCSR(prop);
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::readNetRepo(std::ifstream& fin, Property prop) {
    GraphBase<id_t, off_t>::getNetRepoHeader(fin);
    xlib::UniqueMap<id_t> unique_map;
    std::vector<id2_t> coo_edges_vect(32768);

    size_t n_of_lines = 0;
    while (!fin.eof()) {
        id_t index1, index2;
        fin >> index1;
        fin.ignore(1, ',');
        fin >> index2;
        unique_map.insertValue(index1);
        unique_map.insertValue(index2);

        coo_edges_vect.push_back(std::make_pair(index1 - 1, index2 - 1));
        n_of_lines++;
    }
    _V = static_cast<id_t>(unique_map.size());
    _E = static_cast<off_t>(n_of_lines);
    _coo_size = n_of_lines;
    allocate();
    std::copy(coo_edges_vect.begin(), coo_edges_vect.end(), _coo_edges);
    COOtoCSR(prop);
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::readDimacs10(std::ifstream& fin, Property prop){
    GraphBase<id_t, off_t>::getDimacs10Header(fin);
    _coo_size = static_cast<size_t>(_E);
    allocate();
    xlib::Progress progress(static_cast<size_t>(_V));

    _out_offsets[0] = 0;
    size_t count_edges = 0;
    for (size_t lines = 0; lines < static_cast<size_t>(_V); lines++) {
        std::string str;
        std::getline(fin, str);

        degree_t degree = 0;
        char* token = std::strtok(const_cast<char*>(str.c_str()), " ");
        while (token != nullptr) {
            degree++;
            id_t dest = std::stoi(token) - 1;
            _out_edges[count_edges] = dest;
            _coo_edges[count_edges] = std::make_pair(lines, dest);

            if (_structure.is_directed() && _structure.is_reverse())
                _in_degrees[dest]++;
            count_edges++;
            token = std::strtok(nullptr, " ");
        }
        _out_degrees[lines] = degree;
        if (prop.is_print())
            progress.next(lines);
    }
    assert(count_edges == static_cast<size_t>(_E));
    _out_offsets[0] = 0;
    std::partial_sum(_out_degrees, _out_degrees + _V, _out_offsets + 1);

    if (_structure.is_directed() && _structure.is_reverse()) {
        _in_offsets[0] = 0;
        std::partial_sum(_in_degrees, _in_degrees + _V, _in_offsets + 1);

        auto tmp = new degree_t[_V]();
        for (size_t i = 0; i < static_cast<size_t>(_E); i++) {
            id_t  src = _coo_edges[i].first;
            id_t dest = _coo_edges[i].second;
            _in_edges[ _in_offsets[dest] + tmp[dest]++ ] = src;         //NOLINT
        }
        delete[] tmp;
    }
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::readSnap(std::ifstream& fin, Property prop) {
    _coo_size = GraphBase<id_t, off_t>::getSnapHeader(fin);
    allocate();

    xlib::Progress progress(_coo_size);
    while (fin.peek() == '#')
        xlib::skip_lines(fin);

    xlib::UniqueMap<id_t, id_t> map;
    for (size_t lines = 0; lines < _coo_size; lines++) {
        id_t v1, v2;
        fin >> v1 >> v2;
        _coo_edges[lines] = std::make_pair(map.insertValue(v1),
                                           map.insertValue(v2));
        if (prop.is_print())
            progress.next(lines);
    }
    COOtoCSR(prop);
}

//------------------------------------------------------------------------------

#if defined(__linux__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

template<typename id_t, typename off_t>
void GraphStd<id_t, off_t>::readBinary(const char* filename, Property prop){
    size_t file_size = xlib::file_size(filename);
    xlib::MemoryMapped memory_mapped(filename, file_size,
                                     xlib::MemoryMapped::READ, prop.is_print());

    std::string class_id = xlib::type_name<id_t>() + xlib::type_name<off_t>();
    auto tmp = new char[class_id.size()];
    memory_mapped.read(tmp, class_id.size());

    if (!std::equal(tmp, tmp + class_id.size(), class_id.begin()))
        ERROR("Different class identifier")
    delete[] tmp;

    memory_mapped.read(&_V, 1, &_E, 1, &_structure, 1);
    allocate();

    if (_structure.is_directed() && _structure.is_reverse()) {
        memory_mapped.read(_out_offsets, _V + 1, _in_offsets, _V + 1,   //NOLINT
                           _out_edges, _E, _in_edges, _E);              //NOLINT
        for (id_t i = 0; i < _V; i++)
            _in_degrees[i] = _in_offsets[i + 1] - _in_offsets[i - 1];
    }
    else {
        memory_mapped.read(_out_offsets, _V + 1, _out_edges, _E);       //NOLINT
    }
    for (id_t i = 0; i < _V; i++)
        _out_degrees[i] = _out_offsets[i + 1] - _out_offsets[i - 1];
}

#pragma clang diagnostic pop
#endif
//------------------------------------------------------------------------------

template class GraphStd<int, int>;
template class GraphStd<int64_t, int64_t>;

} // namespace graph
