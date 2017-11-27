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
#include "Graph/WCC.hpp"
#include <iomanip>

namespace graph {

template<typename vid_t, typename eoff_t>
WCC<vid_t, eoff_t>::WCC(const GraphStd<vid_t, eoff_t>& graph) noexcept :
                                            _graph(graph),
                                            _queue(_graph.nV()) {
    _color = new color_t[_graph.nV()];
    std::fill(_color, _color + _graph.nV(), NO_COLOR);
}

template<typename vid_t, typename eoff_t>
WCC<vid_t, eoff_t>::~WCC() noexcept {
    delete[] _color;
}

template<typename vid_t, typename eoff_t>
void WCC<vid_t, eoff_t>::run() noexcept {
    static bool flag = false;
    if (flag)
        ERROR("WCC cannot be repeated")
    flag = true;

    vid_t wcc_count = 0;
    for (vid_t source = 0; source < _graph.nV(); source++) {
        if (_color[source] != NO_COLOR) continue;

        vid_t vertex_count = 0;
        _color[source] = wcc_count;
        _queue.insert(source);

        while (!_queue.empty()) {
            vid_t current = _queue.extract();
            vertex_count++;

            for (eoff_t i = _graph._out_offsets[current];
                 i < _graph._out_offsets[current + 1]; i++) {

                vid_t dest = _graph._out_edges[i];
                if (_color[dest] == NO_COLOR) {
                    _color[dest] = wcc_count;
                    _queue.insert(dest);
                }
            }
            if (!_graph.is_directed()) continue;

            for (eoff_t i = _graph._in_offsets[current];
                i < _graph._in_offsets[current + 1]; i++) {

                vid_t incoming = _graph._in_edges[i];
                if (_color[incoming] == NO_COLOR) {
                    _color[incoming] = true;
                    _queue.insert(incoming);
                }
            }
        }
        _queue.clear();
        _wcc_vector.push_back(vertex_count);
        wcc_count++;
    }
}

template<typename vid_t, typename eoff_t>
vid_t WCC<vid_t, eoff_t>::size() const noexcept {
    return _wcc_vector.size();
}

template<typename vid_t, typename eoff_t>
vid_t WCC<vid_t, eoff_t>::largest() const noexcept {
    return *std::max_element(_wcc_vector.begin(), _wcc_vector.end());
}

template<typename vid_t, typename eoff_t>
vid_t WCC<vid_t, eoff_t>::num_trivial() const noexcept {
    const auto lambda = [](const vid_t& item) { return item == 1; };
    return std::count_if(_wcc_vector.begin(), _wcc_vector.end(), lambda);
}

template<typename vid_t, typename eoff_t>
const std::vector<vid_t>& WCC<vid_t, eoff_t>::list() const noexcept {
    return _wcc_vector;
}

template<typename vid_t, typename eoff_t>
const vid_t* WCC<vid_t, eoff_t>::result() const noexcept {
    return _color;
}

template<typename vid_t, typename eoff_t>
void WCC<vid_t, eoff_t>::print() const noexcept {
    std::cout << "WCCs:\n";
    for (const auto& it : _wcc_vector)
        std::cout << it << " ";
    std::cout << "\n";
}

template<typename vid_t, typename eoff_t>
void WCC<vid_t, eoff_t>::print_histogram() const noexcept {
    vid_t frequency[32] = {};
    for (const auto& it : _wcc_vector)
        frequency[xlib::log2(it)] += it;
    auto log_largest = xlib::log2(largest());
    auto       ratio = xlib::ceil_div<75>(_graph.nV());
    std::cout << "\nWCC Vertices Distribution:\n\n";
    for (int i = 0; i <= log_largest; i++) {
        auto stars = xlib::ceil_div(frequency[i], ratio);
        std::cout << std::left << std::setw(5)
                  << (std::string("2^") + std::to_string(i))
                  << std::string(stars, '*') << "\n";
    }
    std::cout << std::endl;
}

template<typename vid_t, typename eoff_t>
void WCC<vid_t, eoff_t>::print_statistics() const noexcept {
    auto ratio_largest = xlib::per_cent(largest(), _graph.nV());
    auto ratio_trivial = xlib::per_cent(num_trivial(),  _graph.nV());
    std::cout << "\n        Number CC: " << xlib::format(size())
              << "\n       Largest CC: " << xlib::format(ratio_largest, 1)
              << " %"
              << "\n    N. Trivial CC: " << xlib::format(num_trivial())
              << "\n       Trivial CC: " << xlib::format(ratio_trivial, 1)
              << " %" << std::endl;
}
//------------------------------------------------------------------------------

template class WCC<int, int>;
template class WCC<int64_t, int64_t>;

} // namespace graph
