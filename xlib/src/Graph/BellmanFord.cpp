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
#include "Graph/BellmanFord.hpp"

namespace graph {

#define BELLMANFORD BellmanFord<vid_t,eoff_t,weight_t>

template<typename vid_t, typename eoff_t, typename weight_t>
BELLMANFORD::BellmanFord(const GraphWeight<vid_t, eoff_t, weight_t>& graph)
                         noexcept : _graph(graph), _queue(graph.nE()) {
    _distances = new weight_t[_graph._nV];
    reset();
}

template<typename vid_t, typename eoff_t, typename weight_t>
BELLMANFORD::~BellmanFord() noexcept {
    delete[] _distances;
}

template<typename vid_t, typename eoff_t, typename weight_t>
void BELLMANFORD::reset() noexcept {
    std::fill(_distances, _distances + _graph._nV, INF);
    _reset = true;
}

template<typename vid_t, typename eoff_t, typename weight_t>
const weight_t* BELLMANFORD::result() const noexcept {
    return _distances;
}

template<typename vid_t, typename eoff_t, typename weight_t>
void BELLMANFORD::run(vid_t source) noexcept {
    if (!_reset)
        ERROR("BellmanFord not ready")
    const auto& offsets = _graph._out_offsets;

    _queue.insert(source);
    _distances[source] = weight_t(0);

    while (_queue.size() > 0) {
        vid_t next = _queue.extract();
        for (int i = offsets[next]; i < offsets[next + 1]; i++) {
            vid_t dest = _graph._out_edges[i];
            if (relax(next, dest, _graph._out_weights[i]))
                _queue.insert(dest);
        }
    }
    _reset = false;
}

template<typename vid_t, typename eoff_t, typename weight_t>
bool BELLMANFORD::relax(vid_t u, vid_t v, weight_t weight) noexcept {
    if (_distances[u] + weight < _distances[v]) {
        _distances[v] = _distances[u] + weight;
        return true;
    }
    return false;
}

template class BellmanFord<int, int, int>;
template class BellmanFord<int64_t, int64_t, int>;
template class BellmanFord<int, int, float>;
template class BellmanFord<int64_t, int64_t, float>;

} // namespace graph
