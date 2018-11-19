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
#include "Graph/SCC.hpp"
#include "Host/Basic.hpp"
#include <iomanip>

namespace graph {

template<typename vid_t, typename eoff_t>
SCC<vid_t, eoff_t>::SCC(const GraphStd<vid_t, eoff_t>& graph) noexcept :
                                    _graph(graph),
                                    _in_stack(_graph.nV()),
                                    _queue(_graph.nV()) {
    _low_link = new vid_t[_graph.nV()];
    _indices  = new vid_t[_graph.nV()];
    _color    = new color_t[_graph.nV()];
    reset();
}

template<typename vid_t, typename eoff_t>
SCC<vid_t, eoff_t>::~SCC() noexcept {
    delete[] _low_link;
    delete[] _indices;
    delete[] _color;
}

template<typename vid_t, typename eoff_t>
void SCC<vid_t, eoff_t>::reset() noexcept {
    _curr_index = 0;
    _scc_index  = 0;
    std::fill(_low_link, _low_link + _graph.nV(), MAX_LINK);
    std::fill(_indices, _indices + _graph.nV(), NO_INDEX);
    std::fill(_color, _color + _graph.nV(), NO_COLOR);
    _in_stack.clear();
    _queue.clear();
}

template<typename vid_t, typename eoff_t>
void SCC<vid_t, eoff_t>::run() noexcept {
    for (vid_t i = 0; i < _graph.nV(); i++) {
        if (_indices[i] == NO_INDEX) {
            single_scc(i);
            _queue.clear();
        }
    }
}

template<typename vid_t, typename eoff_t>
void SCC<vid_t, eoff_t>::single_scc(vid_t source) noexcept {
    _queue.insert(source);
    _indices[source]  = _low_link[source] = _curr_index++;
    _in_stack[source] = true;

    for (auto i = _graph._out_offsets[source];
         i < _graph._out_offsets[source + 1]; i++) {

        vid_t dest = _graph._out_edges[i];
        if ( _indices[dest] == NO_INDEX ) {
            single_scc(dest);
            _low_link[source] = std::min(_low_link[source], _low_link[dest]);
        }
        else if ( _in_stack[dest] )
            _low_link[source] = std::min(_low_link[source], _indices[dest]);
    }

    if (_indices[source] == _low_link[source]) {
        vid_t extracted;
        vid_t scc_size = 0;
        do {
            scc_size++;
            extracted = _queue.extract();
            _in_stack[extracted] = false;
            _color[extracted] = _scc_index;
        } while (extracted != source);
        _scc_vector.push_back(scc_size);
        _scc_index++;
    }
}

template<typename vid_t, typename eoff_t>
vid_t SCC<vid_t, eoff_t>::size() const noexcept {
    return _scc_vector.size();
}

template<typename vid_t, typename eoff_t>
vid_t SCC<vid_t, eoff_t>::largest() const noexcept {
    return *std::max_element(_scc_vector.begin(), _scc_vector.end());
}

template<typename vid_t, typename eoff_t>
vid_t SCC<vid_t, eoff_t>::num_trivial() const noexcept {
    const auto& lambda = [](const vid_t& item) { return item == 1; };
    return std::count_if(_scc_vector.begin(), _scc_vector.end(), lambda);
}

template<typename vid_t, typename eoff_t>
const std::vector<vid_t>& SCC<vid_t, eoff_t>::list() const noexcept {
    return _scc_vector;
}

template<typename vid_t, typename eoff_t>
const vid_t* SCC<vid_t, eoff_t>::result() const noexcept {
    return _color;
}

template<typename vid_t, typename eoff_t>
void SCC<vid_t, eoff_t>::print() const noexcept {
    std::cout << "SCCs:\n";
    for (const auto& it : _scc_vector)
        std::cout << it << " ";
    std::cout << "\n";
}

template<typename vid_t, typename eoff_t>
void SCC<vid_t, eoff_t>::print_histogram() const noexcept {
    vid_t frequency[32] = {};
    for (const auto& it : _scc_vector)
        frequency[xlib::log2(it)] += it;
    auto log_largest = xlib::log2(largest());
    auto       ratio = xlib::ceil_div<75>(_graph.nV());
    std::cout << "\nSCC Vertices Distribution:\n\n";
    for (int i = 0; i <= log_largest; i++) {
        auto stars = xlib::ceil_div(frequency[i], ratio);
        std::cout << std::left << std::setw(5)
                  << (std::string("2^") + std::to_string(i))
                  << std::string(stars, '*') << "\n";
    }
    std::cout << std::endl;
}
//------------------------------------------------------------------------------

template class SCC<int, int>;
template class SCC<int64_t, int64_t>;

#if defined(STACK)

template<typename vid_t, typename eoff_t>
void SCC<vid_t, eoff_t>::single_scc(vid_t source) noexcept {
        eoff_t i;
        _stack.push({ source, 0 });

NEW_L:  StackNode stack_node = _stack.top();
        source = stack_node.source;

        _queue.insert(source);
        _indices[source] = _low_link[source] = _curr_index++;
        _in_stack.set(source);

        vid_t dest;
        for (i = _graph._out_offsets[source];
             i < _graph._out_offsets[source + 1]; i++) {

            dest = _graph._out_edges[i];
            if ( _indices[dest] == NO_INDEX ) {
                _stack.push({ source, dest, i });
                goto NEW_L;
L_RECURSION:    StackNode stack_node2 = _stack.top();
                source = stack_node.source;
                dest   = stack_node.dest;
                i      = stack_node.i;
                _low_link[source] = std::min(_low_link[source],
                                             _low_link[dest]);
            }
            else if ( _in_stack[dest] )
                _low_link[source] = std::min(_low_link[source], _indices[dest]);
        }

        if (_indices[source] == _low_link[source]) {
            vid_t extracted;
            vid_t scc_size = 0;
            do {
                scc_size++;
                extracted = _queue.extract();
                _in_stack[extracted] = false;
                _color[extracted]    = _scc_index;
            } while (extracted != source);
            _scc_vector.push_back(scc_size);
            _scc_index++;
        }
        _stack.pop();
        goto L_RECURSION;
}

#endif

} // namespace graph
