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
#include "Graph/BFS.hpp"

namespace graph {

template<typename vid_t, typename eoff_t>
BFS<vid_t, eoff_t>::BFS(const GraphStd<vid_t, eoff_t>& graph) noexcept :
                                            _graph(graph),
                                            _bitmask(graph.nV()),
                                            _queue(graph.nV()) {
    _distances = new int[graph.nV()];
    reset();
}

template<typename vid_t, typename eoff_t>
BFS<vid_t, eoff_t>::~BFS() noexcept {
    delete[] _distances;
}

template<typename vid_t, typename eoff_t>
void BFS<vid_t, eoff_t>::run(vid_t source) noexcept {
    if (!_reset)
        ERROR("BFS must be reset before the next run")
    const auto& offsets = _graph._out_offsets;
    _queue.insert(source);
    _bitmask[source]   = true;
    _distances[source] = 0;

    while (!_queue.empty()) {
        auto current = _queue.extract();
        _num_visited++;
        for (eoff_t j = offsets[current]; j < offsets[current + 1]; j++) {
            auto dest = _graph._out_edges[j];
            if (!_bitmask[dest]) {
                _bitmask[dest]   = true;
                _distances[dest] = _distances[current] + 1;
                _queue.insert(dest);
            }
        }
    }
    _reset = false;
}

template<typename vid_t, typename eoff_t>
void BFS<vid_t, eoff_t>::reset() noexcept {
    std::fill(_distances, _distances + _graph.nV(), INF);
    _queue.clear();
    _bitmask.clear();
    _reset       = true;
    _num_visited = 0;
}

template<typename vid_t, typename eoff_t>
vid_t BFS<vid_t, eoff_t>::visited_nodes() const noexcept {
    if (_reset)
        ERROR("BFS not ready")
    return _num_visited;
}

template<typename vid_t, typename eoff_t>
eoff_t BFS<vid_t, eoff_t>::visited_edges() const noexcept {
    if (_reset)
        ERROR("BFS not ready")
    if (_num_visited == _graph.nV())
        return _graph.nE();
    eoff_t sum = 0;
    for (int i = 0; i < _num_visited; i++)
        sum += _graph._out_degrees[ _queue.at(i) ];
    return sum;
}

template<typename vid_t, typename eoff_t>
const typename BFS<vid_t, eoff_t>::dist_t*
BFS<vid_t, eoff_t>::result() const noexcept {
    if (_reset)
        ERROR("BFS not ready")
    return _distances;
}

template<typename vid_t, typename eoff_t>
typename BFS<vid_t, eoff_t>::dist_t
BFS<vid_t, eoff_t>::eccentricity() const noexcept {
    if (_reset)
        ERROR("BFS not ready")
    return _distances[ _queue.tail() ] + 1;
}

template<typename vid_t, typename eoff_t>
std::vector<std::array<vid_t, 4>>
BFS<vid_t, eoff_t>::statistics(vid_t source) noexcept {
    if (!_reset)
        ERROR("BFS must be reset before the next run")
    std::vector<std::array<vid_t, 4>> statistics;
    std::array<vid_t, 4> counter = { { 0, 0, 0, 0 } };

    dist_t      level = 0;
    _distances[source] = 0;
    _queue.insert(source);

    while (!_queue.empty()) {
        vid_t current = _queue.extract();

        if (_distances[current] > level) {
            level++;
            statistics.push_back(counter);
            counter.fill(0);
        }

        const auto& offset = _graph._out_offsets;
        for (eoff_t i = offset[current]; i < offset[current + 1]; i++) {
            vid_t dest = _graph._out_edges[i];

            if (_distances[dest] < level)
                counter[PARENT]++;
            else if (_distances[dest] == level)
                counter[PEER]++;
            else if (_distances[dest] == INF) {
                counter[VALID]++;
                _distances[dest] = level + 1;
                _queue.insert(dest);
            } else
                counter[NOT_VALID]++;
        }
    }
    _reset = false;
    return statistics;
}

template class BFS<int, int>;
template class BFS<int64_t, int64_t>;

} // namespace graph
