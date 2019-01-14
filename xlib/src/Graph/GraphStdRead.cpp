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
#include "Host/Algorithm.hpp"         //xlib::UniqueMap
#include "Host/FileUtil.hpp"          //xlib::skip_lines, xlib::Progress
#include <cstring>                    //std::strtok
#include <sstream>                    //std::istringstream
#include <vector>                     //std::vector

namespace graph {

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readMarket(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getMarketHeader(fin);
    allocate(ginfo);
    xlib::Progress progress(ginfo.num_lines);

    for (size_t lines = 0; lines < ginfo.num_lines; lines++) {
        vid_t index1, index2;
        fin >> index1 >> index2;
        assert(index1 <= _nV && index2 <= _nV);
        _coo_edges[lines] = { index1 - 1, index2 - 1 };

        if (print)
            progress.next(lines);
        xlib::skip_lines(fin);
    }
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readMarketLabel(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getMarketLabelHeader(fin);
    allocate(ginfo);
    xlib::Progress progress(ginfo.num_lines);
    std::string label1, label2;
    xlib::UniqueMap<std::string, vid_t> unique_map;

    for (size_t lines = 0; lines < ginfo.num_lines; lines++) {
        fin >> label1 >> label2;
        vid_t index1 = unique_map.insert(label1);
        vid_t index2 = unique_map.insert(label2);
        _coo_edges[lines] = { index1, index2 };

        if (print)
            progress.next(lines);
        xlib::skip_lines(fin);
    }
    assert(unique_map.size() == ginfo.num_vertices);
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readDimacs9(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getDimacs9Header(fin);
    allocate(ginfo);
    xlib::Progress progress(ginfo.num_lines);

    int c;
    size_t lines = 0;
    while ((c = fin.peek()) != std::char_traits<char>::eof()) {
        if (c == static_cast<int>('a')) {
            vid_t index1, index2;
            xlib::skip_words(fin);
            fin >> index1 >> index2;

            _coo_edges[lines] = { index1 - 1, index2 - 1 };
            if (print)
                progress.next(lines);
            lines++;
        }
        xlib::skip_lines(fin);
    }
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readKonect(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getKonectHeader(fin);
    allocate(ginfo);
    xlib::Progress progress(ginfo.num_lines);

    for (size_t lines = 0; lines < ginfo.num_lines; lines++) {
        vid_t index1, index2;
        fin >> index1 >> index2;
        _coo_edges[lines] = { index1 - 1, index2 - 1 };
        if (print)
            progress.next(lines);
    }
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readNetRepo(std::ifstream& fin) {
    GraphBase<vid_t, eoff_t>::getNetRepoHeader(fin);
    xlib::UniqueMap<vid_t> unique_map;
    std::vector<coo_t> coo_edges_vect(32768);

    size_t num_lines = 0;
    while (!fin.eof()) {
        vid_t index1, index2;
        fin >> index1;
        fin.ignore(1, ',');
        fin >> index2;
        unique_map.insert(index1);
        unique_map.insert(index2);

        coo_edges_vect.push_back( { index1 - 1, index2 - 1 } );
        num_lines++;
    }
    allocate( { unique_map.size(), num_lines, num_lines,
                structure_prop::DIRECTED } );
    std::copy(coo_edges_vect.begin(), coo_edges_vect.end(), _coo_edges);
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readDimacs10(std::ifstream& fin, bool print){
    auto ginfo = GraphBase<vid_t, eoff_t>::getDimacs10Header(fin);
    allocate(ginfo);
    xlib::Progress progress(static_cast<size_t>(ginfo.num_lines));

    size_t count_edges = 0;
    for (size_t lines = 0; lines < ginfo.num_lines; lines++) {
        std::string str;
        std::getline(fin, str);

        char* token = std::strtok(const_cast<char*>(str.c_str()), " ");
        while (token != nullptr) {
            vid_t dest = std::stoi(token) - 1;
            assert(count_edges < ginfo.num_edges);
            _coo_edges[count_edges++] = { lines, dest };
            token = std::strtok(nullptr, " ");
        }
        if (print)
            progress.next(lines);
    }
    //assert(count_edges == ginfo.num_edges);
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readSnap(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getSnapHeader(fin);
    allocate(ginfo);

    xlib::Progress progress(ginfo.num_lines);
    while (fin.peek() == '#')
        xlib::skip_lines(fin);

    xlib::UniqueMap<vid_t, vid_t> map;
    for (size_t lines = 0; lines < ginfo.num_lines; lines++) {
        vid_t v1, v2;
        fin >> v1 >> v2;
        _coo_edges[lines] = { map.insert(v1), map.insert(v2) };
        if (print)
            progress.next(lines);
    }
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readMPG(std::ifstream&, bool) {
    ERROR("readMPG is not valid for GraphStd");
}

//------------------------------------------------------------------------------

#if defined(__linux__)

template<typename vid_t, typename eoff_t>
void GraphStd<vid_t, eoff_t>::readBinary(const char* filename, bool print) {
    size_t file_size = xlib::file_size(filename);
    xlib::MemoryMapped memory_mapped(filename, file_size,
                                     xlib::MemoryMapped::READ, print);

    std::string class_id = xlib::type_name<vid_t>() + xlib::type_name<eoff_t>();
    auto tmp = new char[class_id.size()];
    memory_mapped.read_noprint(tmp, class_id.size());

    if (!std::equal(tmp, tmp + class_id.size(), class_id.begin()))
        ERROR("Different class identifier")
    delete[] tmp;

    memory_mapped.read_noprint(&_nV, 1, &_nE, 1, &_structure, 1);
    auto direction = _structure.is_directed() ? structure_prop::DIRECTED
                                              : structure_prop::UNDIRECTED;
    allocate({static_cast<size_t>(_nV), static_cast<size_t>(_nE),
              static_cast<size_t>(_nE), direction});

    if (_structure.is_directed() && _structure.is_reverse()) {
        memory_mapped.read(_out_offsets, _nV + 1, _in_offsets, _nV + 1, //NOLINT
                           _out_edges, _nE, _in_edges, _nE);            //NOLINT
        for (vid_t i = 0; i < _nV; i++)
            _in_degrees[i] = _in_offsets[i + 1] - _in_offsets[i - 1];
    }
    else
        memory_mapped.read(_out_offsets, _nV + 1, _out_edges, _nE);     //NOLINT

    for (vid_t i = 0; i < _nV; i++)
        _out_degrees[i] = _out_offsets[i + 1] - _out_offsets[i - 1];
    //if (_structure.is_coo())
    //
    std::cout << std::endl;
}

#endif
//------------------------------------------------------------------------------

template class GraphStd<int16_t, int16_t>;
template class GraphStd<int, int>;
template class GraphStd<int64_t, int64_t>;

} // namespace graph
