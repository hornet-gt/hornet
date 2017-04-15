/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
