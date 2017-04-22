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
#include "GraphIO/GraphBase.hpp"
#include "Support/Host/Basic.hpp"   //WARNING
#include "Support/Host/FileUtil.hpp"//xlib::file_size
#include "Support/Host/Numeric.hpp" //xlib::overflowT
#include <iostream>                 //std::cout
#include <sstream>                  //std::istringstream

namespace graph {

template<typename id_t, typename off_t>
void GraphBase<id_t, off_t>::read(const char* filename, Property prop) {//NOLINT
    xlib::check_regular_file(filename);
    size_t size = xlib::file_size(filename);
    _graph_name = xlib::extract_filename(filename);

    if (prop.is_print()) {
        std::cout << "\nReading Graph File...\t" << _graph_name
                  << "       Size: " <<  xlib::format(size >> 20) << " MB";
    }

    std::string file_ext = xlib::extract_file_extension(filename);
    if (file_ext == ".bin") {
        if (prop.is_print())
            std::cout << "            (Binary)\n";
        if (prop.is_randomize() || prop.is_sort())
            std::cerr << "#input sort/randomize ignored on binary format\n";
        readBinary(filename, prop);
        if (prop.is_print())
            print_property();
        return;
    }

    std::ifstream fin;
    //IO improvements START --------------------
    const int BUFFER_SIZE = 1 * xlib::MB;
    char buffer[BUFFER_SIZE];
    //std::ios_base::sync_with_stdio(false);
    fin.tie(nullptr);
    fin.rdbuf()->pubsetbuf(buffer, BUFFER_SIZE);
    //IO improvements END -----------------------
    fin.open(filename);
    std::string first_str;
    fin >> first_str;
    fin.seekg(std::ios::beg);

    if (file_ext == ".mtx" && first_str == "%%MatrixMarket") {
        if (prop.is_print())
            std::cout << "      (MatrixMarket)\n";
        readMarket(fin, prop);
    }
    else if (file_ext == ".graph") {
        if (prop.is_print())
            std::cout << "        (Dimacs10th)\n";
        if (prop.is_randomize() || prop.is_sort()) {
            std::cerr << "#input sort/randomize ignored on Dimacs10th format"
                      << std::endl;
        }
        readDimacs10(fin, prop);
    }
    else if (file_ext == ".gr" && (first_str == "c"|| first_str == "p")) {
        if (prop.is_print())
            std::cout << "         (Dimacs9th)\n";
        readDimacs9(fin, prop);
    }
    else if (file_ext == ".txt" && first_str == "#") {
        if (prop.is_print())
            std::cout << "              (SNAP)\n";
        readSnap(fin, prop);
    }
    else if (file_ext == ".edges") {
        if (prop.is_print())
            std::cout << "    (Net Repository)\n";
        readNetRepo(fin, prop);
    }
    else if (first_str == "%") {
        if (prop.is_print())
            std::cout << "            (Konect)\n";
        readKonect(fin, prop);
    } else
        ERROR("Graph type not recognized");

    fin.close();
    if (prop.is_print())
        print_property();
}

template<typename id_t, typename off_t>
void GraphBase<id_t, off_t>::print_property() {
    assert(_structure.is_direction_set() && _V > 0 && _E > 0);
    const char* graph_dir = _structure.is_undirected()
                                ? "\tGraph Structure: Undirected"
                                : "\tGraph Structure: Directed";
    std::cout << "\nNodes: " << xlib::format(_V) << "\tEdges: "
              << xlib::format(_E) << graph_dir << "\tavg. degree: "
              << xlib::format(static_cast<double>(_E) / _V, 1) << "\n\n";
}

//==============================================================================

template<typename id_t, typename off_t>
size_t GraphBase<id_t, off_t>::getMarketHeader(std::ifstream& fin) {
    std::string header_lines;
    std::getline(fin, header_lines);
    if (!_structure.is_direction_set()) {
        _structure |= header_lines.find("symmetric") != std::string::npos ?
                             Structure::UNDIRECTED : Structure::DIRECTED;
    }
    while (fin.peek() == '%')
        xlib::skip_lines(fin);

    size_t num_lines, rows, columns;
    fin >> rows >> columns >> num_lines;
    if (rows != columns)
        WARNING("Rectangular matrix");

    xlib::overflowT<id_t>(std::max(rows, columns));
    xlib::overflowT<off_t>(num_lines * 2);

    _V = static_cast<id_t>(std::max(rows, columns));
    _E = (_structure.is_undirected()) ? static_cast<off_t>(num_lines) * 2 :
                                        static_cast<off_t>(num_lines);
    xlib::skip_lines(fin);
    return num_lines;
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
size_t GraphBase<id_t, off_t>::getDimacs9Header(std::ifstream& fin) {
    while (fin.peek() == 'c')
        xlib::skip_lines(fin);

    size_t num_lines, n_of_vertices;
    xlib::skip_words(fin, 2);
    fin >> n_of_vertices >> num_lines;
    xlib::overflowT<id_t>(n_of_vertices);
    xlib::overflowT<off_t>(num_lines * 2);

    _structure |= Structure::DIRECTED;
    _V          = static_cast<id_t>(n_of_vertices);
    _E          = static_cast<off_t>(num_lines);
    return num_lines;
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphBase<id_t, off_t>::getDimacs10Header(std::ifstream& fin) {
    while (fin.peek() == '%')
        xlib::skip_lines(fin);

    size_t num_lines, n_of_vertices;
    fin >> n_of_vertices >> num_lines;
    xlib::overflowT<id_t>(n_of_vertices);
    xlib::overflowT<off_t>(num_lines * 2);
    _V = static_cast<id_t>(n_of_vertices);

    if (fin.peek() != '\n' && fin.peek() != ' ') {
        xlib::skip_words(fin, 1);
        _structure |= Structure::DIRECTED;
    } else
        _structure |= Structure::UNDIRECTED;
    xlib::skip_lines(fin);

    _E = (_structure.is_undirected()) ? static_cast<off_t>(num_lines) * 2 :
                                        static_cast<off_t>(num_lines);
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphBase<id_t, off_t>::getKonectHeader(std::ifstream& fin) {
    std::string str;
    fin >> str >> str;
    if (!_structure.is_direction_set()) {
        _structure |= (str == "asym") ? Structure::DIRECTED
                                      : Structure::UNDIRECTED;
    }
    xlib::skip_lines(fin);
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
void GraphBase<id_t, off_t>::getNetRepoHeader(std::ifstream& fin) {
    std::string str;
    fin >> str >> str;
    if (!_structure.is_direction_set()) {
        _structure |= (str == "directed") ? Structure::DIRECTED
                                          : Structure::UNDIRECTED;
    }
    xlib::skip_lines(fin);
}

//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
size_t GraphBase<id_t, off_t>::getSnapHeader(std::ifstream& fin) {
    std::string tmp;
    fin >> tmp >> tmp;
    if (!_structure.is_direction_set()) {
        _structure |= (tmp == "Undirected") ? Structure::UNDIRECTED
                                            : Structure::DIRECTED;
    }
    xlib::skip_lines(fin);

    size_t num_lines = 0, num_vertices = 0;
    while (fin.peek() == '#') {
        std::getline(fin, tmp);
        if (tmp.substr(2, 6) == "Nodes:") {
            std::istringstream stream(tmp);
            stream >> tmp >> tmp >> _V >> tmp >> num_lines;
            break;
        }
    }
    xlib::skip_lines(fin);
    xlib::overflowT<id_t>(_V);
    xlib::overflowT<off_t>(num_lines * 2);
    _V = static_cast<id_t>(num_vertices);
    _E = (_structure.is_undirected()) ? static_cast<off_t>(num_lines) * 2 :
                                        static_cast<off_t>(num_lines);
    return num_lines;
}

//------------------------------------------------------------------------------

template class GraphBase<int, int>;
template class GraphBase<int64_t, int64_t>;

} // namespace graph
