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
 *
 * @file
 */
#pragma once

#include "GraphIO/GraphBase.hpp"
#include <utility>  //std::pair

namespace graph {

template<typename vid_t, typename eoff_t>
class BFS;

template<typename vid_t, typename eoff_t>
class WCC;

template<typename vid_t = int, typename eoff_t = int>
class GraphStd : public GraphBase<vid_t, eoff_t> {
    using    coo_t = typename std::pair<vid_t, vid_t>;
    using degree_t = int;
    friend class BFS<vid_t, eoff_t>;
    friend class WCC<vid_t, eoff_t>;

public:
    class VertexIt;
    class EdgeIt;

    //--------------------------------------------------------------------------
    class Vertex {
        template<typename T, typename R> friend class GraphStd;
    public:
        vid_t     id()         const noexcept;
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
        const vid_t      _id;
        Vertex(vid_t id, const GraphStd& graph) noexcept;
    };

    class VertexIt : public std::iterator<std::forward_iterator_tag, vid_t> {
        template<typename T, typename R> friend class GraphStd;
    public:
        VertexIt& operator++()                   noexcept;
        Vertex    operator*()                    const noexcept;
        bool      operator!=(const VertexIt& it) const noexcept;
    private:
        const GraphStd& _graph;
        eoff_t*         _current;
        explicit VertexIt(eoff_t* current, const GraphStd& graph) noexcept;
    };

    class VerticesContainer {
        template<typename T, typename R> friend class GraphStd;
    public:
        VertexIt begin() const noexcept;
        VertexIt end()   const noexcept;
    private:
        const GraphStd& _graph;

        VerticesContainer(const GraphStd& graph) noexcept;
    };
    //--------------------------------------------------------------------------

    class Edge {
        template<typename T, typename R> friend class GraphStd;
    public:
        eoff_t id()   const noexcept;
        Vertex dest() const noexcept;

        template<typename>
        friend inline std::ostream& operator<<(std::ostream& os,
                                               const Edge& edge) {
            os << edge._id;
            return os;
        }
    private:
        const GraphStd& _graph;
        const eoff_t    _id;

        explicit Edge(eoff_t id, const GraphStd& graph) noexcept;
    };

    class EdgeIt : public std::iterator<std::forward_iterator_tag, vid_t> {
        template<typename T, typename R> friend class GraphStd;
    public:
        EdgeIt& operator++()              noexcept;
        Edge    operator*()               const noexcept;
        bool    operator!=(const EdgeIt& it) const noexcept;
    private:
        const GraphStd& _graph;
        vid_t*          _current;

        explicit EdgeIt(vid_t* current, const GraphStd& graph) noexcept;
    };

    class EdgesContainer {
        template<typename T, typename R> friend class GraphStd;
    public:
        EdgeIt begin() const noexcept;
        EdgeIt end()   const noexcept;
    private:
        const GraphStd& _graph;

        EdgesContainer(const GraphStd& graph) noexcept;
    };
    //--------------------------------------------------------------------------

    /*class InVertexIt :
                        public std::iterator<std::forward_iterator_tag, vid_t> {
        friend class GraphStd<vid_t, eoff_t>::IncomingVerticesContainer;
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

    VerticesContainer V { *this };
    EdgesContainer    E { *this };

    explicit GraphStd()                                    noexcept = default;

    explicit GraphStd(Structure Structure)                 noexcept;

    explicit GraphStd(const char* filename, Property prop) noexcept;

    explicit GraphStd(Structure Structure, const char* filename,
                      Property property) noexcept;

    explicit GraphStd(const eoff_t* csr_offsets, vid_t nV,
                      const vid_t* csr_edges, eoff_t nE) noexcept;

    virtual ~GraphStd() noexcept final;                                 //NOLINT

    Vertex   get_vertex(vid_t index)  const noexcept;
    Edge     get_edge  (eoff_t index) const noexcept;
    degree_t out_degree(vid_t index)  const noexcept;
    degree_t in_degree (vid_t index)  const noexcept;

    const coo_t*    coo_array()   const noexcept;
    const eoff_t*   out_offsets() const noexcept;
    const eoff_t*   in_offsets()  const noexcept;
    const vid_t*    out_edges()   const noexcept;
    const vid_t*    in_edges()    const noexcept;
    const degree_t* out_degrees() const noexcept;
    const degree_t* in_degrees()  const noexcept;

    degree_t  max_out_degree()        const noexcept;
    degree_t  max_in_degree()         const noexcept;
    vid_t     max_out_degree_vertex() const noexcept;
    vid_t     max_in_degree_vertex()  const noexcept;

    bool      is_directed()           const noexcept;

    void print()     const noexcept override;
    void print_raw() const noexcept override;
    void toBinary(const std::string& filename, bool print = true) const;
    void toMarket(const std::string& filename) const;
private:
    eoff_t    *_out_offsets { nullptr };
    eoff_t    *_in_offsets  { nullptr };
    vid_t     *_out_edges   { nullptr };
    vid_t     *_in_edges    { nullptr };
    degree_t* _out_degrees  { nullptr };
    degree_t* _in_degrees   { nullptr };
    coo_t*    _coo_edges    { nullptr };
    size_t    _coo_size     { 0 };
    using GraphBase<vid_t, eoff_t>::_nE;
    using GraphBase<vid_t, eoff_t>::_nV;
    using GraphBase<vid_t, eoff_t>::_structure;

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
