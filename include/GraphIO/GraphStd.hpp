/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "GraphIO/GraphBase.hpp"
#include <utility>  //std::pair

namespace graph {

template<typename id_t = int, typename off_t = int>
class GraphStd : public GraphBase<id_t, off_t> {
using    id2_t = typename std::pair<id_t, id_t>;
using degree_t = int;

public:
    class VertexIt;
    class EdgeIt;

    //--------------------------------------------------------------------------
    class Vertex {
        template<typename T, typename R> friend class GraphStd;
    public:
        id_t     id()         const noexcept;
        degree_t out_degree() const noexcept;
        degree_t in_degree()  const noexcept;

        friend inline std::ostream& operator<<(std::ostream& os,
                                               const Vertex& vertex) {
            os << vertex._id;
            return os;
        }

        EdgeIt begin()  const noexcept;
        EdgeIt end()    const noexcept;
    private:
        const GraphStd& _graph;
        const id_t      _id;
        Vertex(id_t id, const GraphStd& graph) noexcept;
    };

    class VertexIt : public std::iterator<std::forward_iterator_tag, id_t> {
        template<typename T, typename R> friend class GraphStd;
    public:
        VertexIt& operator++()                   noexcept;
        Vertex    operator*()                    const noexcept;
        bool      operator!=(const VertexIt& it) const noexcept;
    private:
        const GraphStd& _graph;
        off_t*          _current;
        explicit VertexIt(off_t* current, const GraphStd& graph) noexcept;
    };

    class VerticesContainer {
        template<typename T, typename R> friend class GraphStd;
    public:
        VertexIt begin() const noexcept;
        VertexIt end()   const noexcept;

        VerticesContainer(const VerticesContainer&) = delete;
    private:
        const GraphStd& _graph;

        explicit VerticesContainer(const GraphStd& graph) noexcept;
    };
    //--------------------------------------------------------------------------

    class Edge {
        template<typename T, typename R> friend class GraphStd;
    public:
        off_t  id()                  const noexcept;
        Vertex dest()                const noexcept;

        template<typename>
        friend inline std::ostream& operator<<(std::ostream& os,
                                               const Edge& edge) {
            os << edge._id;
            return os;
        }
    private:
        const GraphStd& _graph;
        const off_t     _id;

        explicit Edge(off_t id, const GraphStd& graph) noexcept;
    };

    class EdgeIt : public std::iterator<std::forward_iterator_tag, id_t> {
        template<typename T, typename R> friend class GraphStd;
    public:
        EdgeIt& operator++()              noexcept;
        Edge    operator*()               const noexcept;
        bool operator!=(const EdgeIt& it) const noexcept;
    private:
        const GraphStd& _graph;
        id_t* _current;

        explicit EdgeIt(id_t* current, const GraphStd& graph) noexcept;
    };

    class EdgesContainer {
        template<typename T, typename R> friend class GraphStd;
    public:
        EdgeIt begin() const noexcept;
        EdgeIt end()   const noexcept;

        EdgesContainer(const EdgesContainer&) = delete;
    private:
        const GraphStd& _graph;

        explicit EdgesContainer(const GraphStd& graph)        noexcept;
    };
    //--------------------------------------------------------------------------

    /*class InVertexIt :
                        public std::iterator<std::forward_iterator_tag, id_t> {
        friend class GraphStd<id_t, off_t>::IncomingVerticesContainer;
    public:
        InVertexIt& operator++()                   noexcept;
        IncomingVertex    operator*()                    const noexcept;
        bool      operator!=(const InVertexIt& it) const noexcept;

        void operator=(const InVertexIt&) = delete;
    private:
        const GraphStd& _graph;
        id-t*           _current;
        explicit InVertexIt(const GraphStd& graph) noexcept;
    };

    class IncomingVertex {
    public:
        InVertexIt begin() const noexcept;
        InVertexIt end()   const noexcept;

        Incoming(const Incoming&) = delete;
        Incoming& operator=(const Incoming&& obj) = delete;
    private:
        const GraphStd& _graph;
        explicit Incoming(const GraphStd& graph) noexcept;
    };*/
    //==========================================================================

    VerticesContainer V;
    EdgesContainer    E;

    explicit GraphStd()                                    noexcept;
    explicit GraphStd(Structure Structure)                 noexcept;
    explicit GraphStd(const char* filename, Property prop) noexcept;
    explicit GraphStd(Structure Structure, const char* filename,
                      Property property) noexcept;
    virtual ~GraphStd() noexcept;

    Vertex   get_vertex(id_t index)  const noexcept;
    Edge     get_edge  (off_t index) const noexcept;
    degree_t out_degree(id_t index)  const noexcept;
    degree_t in_degree (id_t index)  const noexcept;

    const id2_t*    coo_array()         const noexcept;
    const off_t*    out_offsets_array() const noexcept;
    const off_t*    in_offsets_array()  const noexcept;
    const id_t*     out_edges_array()   const noexcept;
    const id_t*     in_edges_array()    const noexcept;
    const degree_t* out_degrees_array() const noexcept;
    const degree_t* in_degrees_array()  const noexcept;

    void print()     const noexcept override;
    void print_raw() const noexcept override;
    /**
     * @warning out_degree/in_degree not store (performance reason)
     */
    void toBinary(const std::string& filename, bool print = true) const;
    void toMarket(const std::string& filename, bool print = true) const;
private:
    off_t     *_out_offsets { nullptr };
    off_t     *_in_offsets  { nullptr };
    id_t      *_out_edges   { nullptr };
    id_t      *_in_edges    { nullptr };
    degree_t* _out_degrees  { nullptr };
    degree_t* _in_degrees   { nullptr };
    id2_t*    _coo_edges    { nullptr };
    size_t    _coo_size     { 0 };
    using GraphBase<id_t, off_t>::_E;
    using GraphBase<id_t, off_t>::_V;
    using GraphBase<id_t, off_t>::_structure;

    void allocate() noexcept override;

    void readMarket  (std::ifstream& fin, Property prop)   override;
    void readDimacs9 (std::ifstream& fin, Property prop)   override;
    void readDimacs10(std::ifstream& fin, Property prop)   override;
    void readSnap    (std::ifstream& fin, Property prop)   override;
    void readKonect  (std::ifstream& fin, Property prop)   override;
    void readNetRepo (std::ifstream& fin, Property prop)   override;
    void readBinary  (const char* filename, Property prop) override;

    void COOtoCSR(Property prop) noexcept;
    void CSRtoCOO(Property prop) noexcept;
};

} // namespace graph

#include "GraphStd.i.hpp"
