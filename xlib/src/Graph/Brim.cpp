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
#include "Graph/Brim.hpp"
#include "Host/FileUtil.hpp"
#include <fstream>

namespace graph {

template<typename vid_t, typename eoff_t, typename weight_t>
Brim<vid_t, eoff_t, weight_t>
::Brim(const GraphWeight<vid_t, eoff_t, weight_t>& graph) noexcept :
                                            _graph(graph),
                                            _in_queue(_graph.nV()),
                                            _queue(_graph.nV()) {
    _potentials = new potential_t[_graph.nV()];
    _counters   = new potential_t[_graph.nV()];
    reset();
}

template<typename vid_t, typename eoff_t, typename weight_t>
Brim<vid_t, eoff_t, weight_t>::~Brim() noexcept {
    delete[] _potentials;
    delete[] _counters;
}

template<typename vid_t, typename eoff_t, typename weight_t>
void Brim<vid_t, eoff_t, weight_t>::reset() noexcept {
    const auto& offsets = _graph._out_offsets;
    std::fill(_potentials, _potentials + _graph.nV(), 0);
    std::fill(_counters, _counters + _graph.nV(), 0);

    for (auto i = 0; i < _graph.nV(); i++) {
        if (is_player0(i)) {
            int count = 0;
            for (auto j = offsets[i]; j < offsets[i + 1]; j++) {
                if (_graph._out_weights[j] >= weight_t(0))
                    count++;
            }
            if (count == 0) {
                _queue.insert(i);
                _in_queue[i] = true;
            }
            _counters[i] = count;
        }
        else {
            for (auto j = offsets[i]; j < offsets[i + 1]; j++) {
                if (_graph._out_weights[j] < 0) {
                    _queue.insert(i);
                    _in_queue[i] = true;
                    break;
                }
            }
        }
    }
    findMg();
}

//==============================================================================

template<typename vid_t, typename eoff_t, typename weight_t>
void Brim<vid_t, eoff_t, weight_t>::run() noexcept {
	while (_queue.size() > 0) {
        const auto& in_offsets = _graph._in_offsets;

		auto       vertex_id = _queue.extract();
		_in_queue[vertex_id] = false;
		auto   old_potential = _potentials[vertex_id];
		lift_count(vertex_id);

		if (_potentials[vertex_id] == TT)
			return;

		for (auto j = in_offsets[vertex_id];
             j < in_offsets[vertex_id + 1]; j++) {

            auto in_weight = _graph._in_weights[j];
            auto  incoming = _graph._in_edges[j];

			if (_potentials[incoming] < minus(_potentials[vertex_id], in_weight)) {
				if (is_player0(incoming)) {
					if (_potentials[incoming] >= minus(old_potential, in_weight)) {
						_counters[incoming]--;

						if (_counters[incoming] <= 0 && !_in_queue[incoming]) {
							_queue.insert(incoming);
							_in_queue[incoming] = true;
						}
					}
				}
				else if (!_in_queue[incoming]) {
					_queue.insert(incoming);
					_in_queue[incoming] = true;
				}
			}
		}
	}
}

/*
void GraphWeight::run2() {
	while (_queue.size() > 0) {
		auto size = _queue.size();

		for (auto i = 0; i < size; i++) {
			int vertex_id = _queue.get(i);
			_tmp_potential[vertex_id] = _potentials[vertex_id];
			lift_count(vertex_id);

			if (_potentials[vertex_id] == TT)
				return;
		}

		for (auto i = 0; i < size; i++) {
			auto vertex_id = _queue.extract();

			_in_queue[vertex_id] = false;
			auto   old_potential = _tmp_potential[vertex_id];

			for (int j = InNodes[vertex_id]; j < InNodes[vertex_id + 1]; j++) {
                auto in_weight = _graph._in_weights[j];
                auto  incoming = _graph._in_edges[j];

				if (_potentials[incoming] < minus(_potentials[vertex_id],
                                                 in_weight)) {
					if (is_player0(incoming)) {
						if (_potentials[incoming] >= minus(old_potential,
                                                          in_weight)) {
							_counters[incoming]--;

							if (_counters[incoming] <= 0 &&
                                !_in_queue[incoming]) {

								_queue.insert(incoming);
								_in_queue[incoming] = true;
							}
						}
					}
					else if (!_in_queue[incoming]) {
						_queue.insert(incoming);
						_in_queue[incoming] = true;
					}
				}
			}
		}
	}
}*/

//==============================================================================

template<typename vid_t, typename eoff_t, typename weight_t>
void Brim<vid_t, eoff_t, weight_t>::findMg() noexcept {
    const auto& offsets = _graph._out_offsets;
	Mg = 0;
	for (auto i = 0; i < _graph.nV(); i++) {
		potential_t max = 0;
		for (auto j = offsets[i]; j < offsets[i + 1]; j++)
			max = std::max(max, -(_graph._out_weights[j]));
		Mg += max;
	}
	std::cout << "MG: " << Mg << std::endl;

	potential_t W = std::numeric_limits<potential_t>::lowest();
	for (auto i = 0; i < _graph.nE(); i++)
		W = std::max(W, std::abs(_graph._out_weights[i]));
	if (Mg > _graph.nV() * W)
		ERROR("Mg > V * W  : ", Mg, " > ", _graph.nV(), " * ", W)
}

template<typename vid_t, typename eoff_t, typename weight_t>
void Brim<vid_t, eoff_t, weight_t>::lift_count(vid_t vertex_id) noexcept {
    const auto& offsets = _graph._out_offsets;

    if (is_player0(vertex_id)) {
		auto min_value = std::numeric_limits<potential_t>::max();
		int count = 0;
		for (int j = offsets[vertex_id]; j < offsets[vertex_id + 1]; j++) {
            auto weight = _graph._out_weights[j];
            auto    dst = _graph._out_edges[j];
			auto   diff = minus(_potentials[dst], weight);
			if (diff < min_value) {
				min_value = diff;
				count     = 1;
			}
			else if (diff == min_value)
				count++;
		}
		_potentials[vertex_id] = min_value;
		_counters[vertex_id]  = count;
	}
	else {
		auto max_value = std::numeric_limits<potential_t>::lowest();
		for (int j = offsets[vertex_id]; j < offsets[vertex_id + 1]; j++) {
            auto weight = _graph._out_weights[j];
            auto    dst = _graph._out_edges[j];
			max_value = std::max(max_value, minus(_potentials[dst], weight));
		}
		_potentials[vertex_id] = max_value;
	}
	if (_potentials[vertex_id] > Mg)
		_potentials[vertex_id] = TT;
}

template<typename vid_t, typename eoff_t, typename weight_t>
auto Brim<vid_t, eoff_t, weight_t>::minus(potential_t a, potential_t b)
                                          const noexcept {
	if (a != TT && a - b <= Mg)
		return std::max(potential_t(0), a - b);
	return TT;
}

template<typename vid_t, typename eoff_t, typename weight_t>
void Brim<vid_t, eoff_t, weight_t>::set_player_TH(vid_t index) noexcept {
	_player_TH = index;
}

template<typename vid_t, typename eoff_t, typename weight_t>
bool Brim<vid_t, eoff_t, weight_t>::is_player0(vid_t index) const noexcept {
	return index >= _player_TH;
}

//==============================================================================


template<typename vid_t, typename eoff_t, typename weight_t>
const typename Brim<vid_t, eoff_t, weight_t>::potential_t*
Brim<vid_t, eoff_t, weight_t>::result() const noexcept {
    return _potentials;
}

template<typename vid_t, typename eoff_t, typename weight_t>
bool Brim<vid_t, eoff_t, weight_t>::check() const noexcept {
    const auto& offsets = _graph._out_offsets;

	for (auto i = 0; i < _graph.nV(); i++) {
		if (is_player0(i)) {
			bool flag = true;
			for (auto j = offsets[i]; j < offsets[i + 1]; j++) {
                auto weight = _graph._out_weights[j];
                auto    dst = _graph._out_edges[j];
				if (_potentials[i] >= minus(_potentials[dst], weight))
					flag = false;
			}
			if (flag) {
				std::cerr << "(A) error _potentials : " << _potentials[i]
                          << " at position " << i << std::endl;
				return false;
			}
		}
		else {
			for (auto j = offsets[i]; j < offsets[i + 1]; j++) {
                auto weight = _graph._out_weights[j];
                auto    dst = _graph._out_edges[j];
				if (_potentials[i] < minus(_potentials[dst], weight)) {
					std::cout << "(B) error _potentials : " << _potentials[i]
                              << " at position " << i << std::endl;
					return false;
				}
			}
		}
	}
	return true;
}

template<typename vid_t, typename eoff_t, typename weight_t>

void Brim<vid_t, eoff_t, weight_t>::print_potential() const noexcept {
	std::cout << "Potential:\n";
	for (auto i = 0; i < _graph.nV(); i++) {
		if (_potentials[i] == TT)
			std::cout << "T ";
		else
			std::cout << _potentials[i] << " ";
	}
	std::cout << std::endl;
}

template<typename vid_t, typename eoff_t, typename weight_t>
void Brim<vid_t, eoff_t, weight_t>::print_potential_to_file() const {
	std::ofstream result("result.txt");
	for (auto i = 0; i < _graph.nV(); i++)
		result << "vertex_id: " << i << " _potentials: " << _potentials[i] << "\n";
	result << std::endl;
	result.close();
}

template<typename vid_t, typename eoff_t, typename weight_t>
void Brim<vid_t, eoff_t, weight_t>::check_from_file(const std::string& file) {
	std::ifstream fin(file);
	std::string str;
	for (auto i = 0; i < _graph.nV(); i++) {
		potential_t value;
		fin >> str >> str >> str >> value;
		xlib::skip_lines(fin);

		if (_potentials[i] != value) {
			std::cerr << "\n** ERROR  " << i << " _potentials: " << _potentials[i]
                      << " f:" << value << "\n\n";
			return;
		}
        str.clear();
	}
	std::cout << "Correct <>" << std::endl;
}

//==============================================================================

template class Brim<int, int, int>;
template class Brim<int64_t, int64_t, int>;

} // namespace graph
