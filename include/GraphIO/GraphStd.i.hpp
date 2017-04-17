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
#include <cassert>

namespace graph {

////////////////////////////////
///         Vertex           ///
////////////////////////////////
template<typename id_t, typename off_t>
inline GraphStd<id_t, off_t>
::Vertex::Vertex(id_t id, const GraphStd& graph) noexcept : _graph(graph),
                                                            _id(id) {};

template<typename id_t, typename off_t>
inline id_t GraphStd<id_t, off_t>::Vertex::id() const noexcept {
    return _id;
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::degree_t
GraphStd<id_t, off_t>::Vertex::out_degree() const noexcept {
    return _graph._out_degrees[_id];
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::degree_t
GraphStd<id_t, off_t>::Vertex::in_degree() const noexcept {
    return _graph._in_degrees[_id];
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::EdgeIt
GraphStd<id_t, off_t>::Vertex::begin() const noexcept {
    return EdgeIt(_graph._out_edges + _graph._out_offsets[_id], _graph);
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::EdgeIt
GraphStd<id_t, off_t>::Vertex::end() const noexcept {
    return EdgeIt(_graph._out_edges + _graph._out_offsets[_id + 1], _graph);
}
//==============================================================================
////////////////////////////////
///         VertexIt         ///
////////////////////////////////
template<typename id_t, typename off_t>
inline GraphStd<id_t, off_t>::VertexIt
::VertexIt(off_t* current, const GraphStd& graph) noexcept :
                    _graph(graph), _current(current) {}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::VertexIt&
GraphStd<id_t, off_t>::VertexIt::VertexIt::operator++ () noexcept {
    _current++;
    return *this;
}

template<typename id_t, typename off_t>
inline bool
GraphStd<id_t, off_t>::VertexIt::operator!= (const VertexIt& it)
                                        const noexcept {
    return _current != it._current;
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::Vertex
GraphStd<id_t, off_t>::VertexIt::operator* () const noexcept {
    return Vertex(static_cast<id_t>(_current - _graph._out_offsets), _graph);
}
//==============================================================================
////////////////////////////////
///         Edge             ///
////////////////////////////////
template<typename id_t, typename off_t>
inline GraphStd<id_t, off_t>
::Edge::Edge(off_t id, const GraphStd& graph) noexcept : _graph(graph),
                                                         _id(id) {};

template<typename id_t, typename off_t>
inline off_t GraphStd<id_t, off_t>::Edge::id() const noexcept {
    return _id;
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::Vertex
GraphStd<id_t, off_t>::Edge::dest() const noexcept {
    return Vertex(_graph._out_edges[_id], _graph);
}
//==============================================================================
////////////////////////////////
///         EdgeIt           ///
////////////////////////////////
template<typename id_t, typename off_t>
inline GraphStd<id_t, off_t>::EdgeIt
::EdgeIt(id_t* current, const GraphStd& graph) noexcept :
                           _graph(graph), _current(current) {}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::EdgeIt&
GraphStd<id_t, off_t>::EdgeIt::EdgeIt::operator++() noexcept {
    _current++;
    return *this;
}

template<typename id_t, typename off_t>
inline bool
GraphStd<id_t, off_t>::EdgeIt::operator!=(const EdgeIt& it) const noexcept {
    return _current != it._current;
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::Edge
GraphStd<id_t, off_t>::EdgeIt::operator*() const noexcept {
    return Edge(static_cast<id_t>(_current - _graph._out_edges), _graph);
}

//==============================================================================
////////////////////////////////
///  VerticesContainer       ///
////////////////////////////////
template<typename id_t, typename off_t>
inline GraphStd<id_t, off_t>::VerticesContainer
::VerticesContainer(const GraphStd& graph) noexcept : _graph(graph) {}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::VertexIt
GraphStd<id_t, off_t>::VerticesContainer::begin() const noexcept {
    return VertexIt(_graph._out_offsets, _graph);
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::VertexIt
GraphStd<id_t, off_t>::VerticesContainer::end() const noexcept {
    return VertexIt(_graph._out_offsets + _graph._V, _graph);
}

//==============================================================================
////////////////////////////////
///     EdgesContainer       ///
////////////////////////////////
template<typename id_t, typename off_t>
inline GraphStd<id_t, off_t>::EdgesContainer
::EdgesContainer(const GraphStd& graph) noexcept : _graph(graph) {}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::EdgeIt
GraphStd<id_t, off_t>::EdgesContainer::begin() const noexcept {
    return EdgeIt(_graph._out_edges, _graph);
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::EdgeIt
GraphStd<id_t, off_t>::EdgesContainer::end() const noexcept {
    return EdgeIt(_graph._out_edges + _graph._E, _graph);
}

//==============================================================================
////////////////////////////////
///         GRAPHSTD         ///
////////////////////////////////

template<typename id_t, typename off_t>
inline const off_t* GraphStd<id_t, off_t>::out_offsets_array() const noexcept {
    return _out_offsets;
}

template<typename id_t, typename off_t>
inline const off_t* GraphStd<id_t, off_t>::in_offsets_array() const noexcept {
    return _in_offsets;
}

template<typename id_t, typename off_t>
inline const id_t* GraphStd<id_t, off_t>::out_edges_array() const noexcept {
    return _out_edges;
}

template<typename id_t, typename off_t>
inline const id_t* GraphStd<id_t, off_t>::in_edges_array() const noexcept {
    return _in_edges;
}

template<typename id_t, typename off_t>
inline const typename GraphStd<id_t, off_t>::degree_t*
GraphStd<id_t, off_t>::out_degrees_array() const noexcept{
    return _out_degrees;
}

template<typename id_t, typename off_t>
inline const typename GraphStd<id_t, off_t>::degree_t*
GraphStd<id_t, off_t>::in_degrees_array() const noexcept{
    return _in_degrees;
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::degree_t
GraphStd<id_t, off_t>::out_degree(id_t index) const noexcept{
    assert(index >= 0 && index < _V);
    return _out_degrees[index];
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::degree_t
GraphStd<id_t, off_t>::in_degree(id_t index) const noexcept{
    assert(index >= 0 && index < _V);
    return _in_degrees[index];
}

template<typename id_t, typename off_t>
inline typename GraphStd<id_t, off_t>::Vertex
GraphStd<id_t, off_t>::get_vertex(id_t index) const noexcept {
    return Vertex(index, *this);
}

} //namespace graph
