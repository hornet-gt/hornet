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
#include "GraphIO/WCC.hpp"

namespace graph {

template<typename vid_t, typename eoff_t>
WCC<vid_t, eoff_t>::WCC(const GraphStd<vid_t, eoff_t>& graph) noexcept :
                                            _graph(graph),
                                            _bitmask(_graph.nV()),
                                            _queue(_graph.nV()) {
    if (_graph.is_directed())
        ERROR("The graph must be directed")
}

template<typename vid_t, typename eoff_t>
void WCC<vid_t, eoff_t>::run() noexcept {
    static bool flag = false;
    if (flag)
        ERROR("WCC cannot be repeated")
    flag = true;

    for (vid_t source = 0; source < _graph.nV(); source++) {
        if (_bitmask[source]) continue;

        vid_t count = 0;
        _bitmask[source] = true;
        _queue.insert(source);

        while (!_queue.is_empty()) {
            vid_t current = _queue.extract();
            count++;

            for (eoff_t i = _graph._out_offsets[current];
                 i < _graph._out_offsets[current + 1]; i++) {

                vid_t dest = _graph._out_edges[i];
                if (!_bitmask[dest]) {
                    _bitmask[dest] = true;
                    _queue.insert(dest);
                }
            }
            for (eoff_t i = _graph._in_offsets[current];
                i < _graph._in_offsets[current + 1]; i++) {

                vid_t incoming = _graph._in_edges[i];
                if (!_bitmask[incoming]) {
                    _bitmask[incoming] = true;
                    _queue.insert(incoming);
                }
            }
        }
        _wcc_vector.push_back(count);
    }
}

template<typename vid_t, typename eoff_t>
vid_t WCC<vid_t, eoff_t>::size() const noexcept {
    return _wcc_vector.size();
}

template<typename vid_t, typename eoff_t>
vid_t WCC<vid_t, eoff_t>::largest_size() const noexcept {
    return _wcc_vector.back();
}

template<typename vid_t, typename eoff_t>
vid_t WCC<vid_t, eoff_t>::num_trivial() const noexcept {
    const auto lambda = [](const vid_t& item) { return item == 0; };
    return std::count_if(_wcc_vector.begin(), _wcc_vector.end(), lambda);
}

template<typename vid_t, typename eoff_t>
const std::vector<vid_t>& WCC<vid_t, eoff_t>::vector() const noexcept {
    return _wcc_vector;
}

//------------------------------------------------------------------------------

template class WCC<int, int>;
template class WCC<int64_t, int64_t>;

} // namespace graph
