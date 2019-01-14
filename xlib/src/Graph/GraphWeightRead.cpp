/**
 * @internal
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
 *
 * @file
 */
#include "Graph/GraphWeight.hpp"
#include "Host/Algorithm.hpp" //xlib::UniqueMap
#include "Host/FileUtil.hpp"  //xlib::skip_lines, xlib::Progress
#include <cstring>            //std::strtok
#include <sstream>            //std::istringstream
#include <vector>             //std::vector

namespace graph {

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readMarket(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getMarketHeader(fin);
    allocate(ginfo);
    xlib::Progress progress(_coo_size);

    for (size_t lines = 0; lines < _coo_size; lines++) {
        vid_t index1, index2;
        weight_t weight;
        fin >> index1 >> index2 >> weight;
        _coo_edges[lines] = coo_t(index1 - 1, index2 - 1, weight);

        if (print)
            progress.next(lines);
        xlib::skip_lines(fin);
    }
}

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readDimacs9(std::ifstream& fin, bool print) {
    ERROR("Not Implemented")
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readKonect(std::ifstream& fin, bool print) {
    ERROR("Not Implemented")
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readNetRepo(std::ifstream& fin) {
    ERROR("Not Implemented")
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readDimacs10(std::ifstream& fin, bool print) {
    ERROR("Not Implemented")
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readSnap(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getSnapHeader(fin);
    allocate(ginfo);

    xlib::Progress progress(_coo_size);
    while (fin.peek() == '#')
        xlib::skip_lines(fin);

    xlib::UniqueMap<vid_t, vid_t> map;
    for (size_t lines = 0; lines < _coo_size; lines++) {
        vid_t v1, v2;
        weight_t weight;
        fin >> v1 >> v2 >> weight;
        _coo_edges[lines] = coo_t(map.insert(v1), map.insert(v2), weight);
        if (print)
            progress.next(lines);
    }
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readMPG(std::ifstream& fin, bool print) {
    auto ginfo = GraphBase<vid_t, eoff_t>::getMPGHeader(fin);
    allocate(ginfo);

    /// TO IMPLEMENT
}

//------------------------------------------------------------------------------

#if defined(__linux__)

template<typename vid_t, typename eoff_t, typename weight_t>
void GraphWeight<vid_t, eoff_t, weight_t>
::readBinary(const char* filename, bool print) {
    size_t file_size = xlib::file_size(filename);
    xlib::MemoryMapped memory_mapped(filename, file_size,
                                     xlib::MemoryMapped::READ, print);

    std::string class_id = xlib::type_name<vid_t>() + xlib::type_name<eoff_t>() +
                           xlib::type_name<weight_t>();
    auto tmp = new char[class_id.size()];
    memory_mapped.read(tmp, class_id.size());

    if (!std::equal(tmp, tmp + class_id.size(), class_id.begin()))
        ERROR("Different class identifier")
    delete[] tmp;

    memory_mapped.read(&_nV, 1, &_nE, 1, &_structure, 1);
    auto direction = _structure.is_directed() ? structure_prop::DIRECTED
                                              : structure_prop::UNDIRECTED;
    allocate({static_cast<size_t>(_nV), static_cast<size_t>(_nE),
              static_cast<size_t>(_nE), direction});

    if (_structure.is_directed() && _structure.is_reverse()) {
        memory_mapped.read(_out_offsets, _nV + 1, _in_offsets, _nV + 1, //NOLINT
                           _out_edges, _nE, _in_edges, _nE,             //NOLINT
                           _out_weights, _nE, _in_weights, _nE);
        for (vid_t i = 0; i < _nV; i++)
            _in_degrees[i] = _in_offsets[i + 1] - _in_offsets[i - 1];
    }
    else {
        memory_mapped.read(_out_offsets, _nV + 1, _out_edges, _nE,      //NOLINT
                           _out_weights, _nE);                          //NOLINT
    }
    for (vid_t i = 0; i < _nV; i++)
        _out_degrees[i] = _out_offsets[i + 1] - _out_offsets[i - 1];
}

#endif
//------------------------------------------------------------------------------

template class GraphWeight<int, int, int>;
template class GraphWeight<int, int, float>;
template class GraphWeight<int64_t, int64_t, int64_t>;

} // namespace graph
