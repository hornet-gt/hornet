/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date June, 2017
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

#include "Host/Basic.hpp"   //xlib::PropertyClass
#include <string>                   //std::string

namespace graph {

namespace detail {
    enum class ParsingEnum { RANDOMIZE = 1, SORT = 2, REMOVE_DUPLICATES = 4,
                             PRINT = 8 };
} // namespace detail

class ParsingProp : public xlib::PropertyClass<detail::ParsingEnum,
                                               ParsingProp> {
    template<typename, typename>           friend class GraphBase;
    template<typename, typename>           friend class GraphStd;
    template<typename, typename, typename> friend class GraphWeight;
public:
    explicit ParsingProp() noexcept = default;
    explicit ParsingProp(const detail::ParsingEnum& value) noexcept;
private:
    bool is_sort()              const noexcept;
    bool is_randomize()         const noexcept;
    bool is_remove_duplicates() const noexcept;
    bool is_print()             const noexcept;
};

namespace parsing_prop {

const ParsingProp         RANDOMIZE( detail::ParsingEnum::RANDOMIZE );
const ParsingProp              SORT( detail::ParsingEnum::SORT );
const ParsingProp REMOVE_DUPLICATES( detail::ParsingEnum::REMOVE_DUPLICATES );
const ParsingProp             PRINT( detail::ParsingEnum::PRINT );

} // namespace parsing_prop

//==============================================================================
namespace detail {
    enum class StructureEnum { DIRECTED = 1, UNDIRECTED = 2, REVERSE = 4,
                               COO = 8 };
} // namespace detail

class StructureProp :
              public xlib::PropertyClass<detail::StructureEnum, StructureProp> {
    template<typename, typename>           friend class GraphBase;
    template<typename, typename>           friend class GraphStd;
    template<typename, typename, typename> friend class GraphWeight;
public:
    explicit StructureProp() noexcept = default;
    explicit StructureProp(const detail::StructureEnum& value) noexcept;
private:
    //enum WType   { NONE, INTEGER, REAL };
    //WType _wtype { NONE };
    bool is_directed()      const noexcept;
    bool is_undirected()    const noexcept;
    bool is_reverse()       const noexcept;
    bool is_coo()           const noexcept;
    bool is_direction_set() const noexcept;
    bool is_weighted()      const noexcept;
};

namespace structure_prop {

const StructureProp DIRECTED  ( detail::StructureEnum::DIRECTED );
const StructureProp UNDIRECTED( detail::StructureEnum::UNDIRECTED );
const StructureProp REVERSE   ( detail::StructureEnum::REVERSE );
const StructureProp COO       ( detail::StructureEnum::COO );

} // namespace structure_prop

//==============================================================================

struct GInfo {
    size_t        num_vertices;
    size_t        num_edges;
    size_t        num_lines;
    StructureProp direction;
};

template<typename vid_t, typename eoff_t>
class GraphBase {
public:
    virtual vid_t  nV() const noexcept final;
    virtual eoff_t nE() const noexcept final;
    virtual const std::string& name() const noexcept final;

    virtual void read(const char* filename,
                      const ParsingProp& prop =
                            ParsingProp(parsing_prop::PRINT)) final;    //NOLINT

    virtual void print()     const noexcept = 0;
    virtual void print_raw() const noexcept = 0;

    GraphBase(const GraphBase&)      = delete;
    void operator=(const GraphBase&) = delete;
protected:
    StructureProp _structure;
    ParsingProp   _prop;
    std::string   _graph_name { "" };
    vid_t         _nV         { 0 };
    eoff_t        _nE         { 0 };
    bool          _directed_to_undirected { false };
    bool          _undirected_to_directed { false };
    bool          _stored_undirected      { false };

    explicit GraphBase() = default;
    explicit GraphBase(StructureProp structure) noexcept;
    explicit GraphBase(vid_t nV, eoff_t nE, StructureProp structure)
                       noexcept;
    virtual ~GraphBase() noexcept = default;

    virtual void   set_structure(const StructureProp& structure) noexcept final;

    virtual void   readMarket   (std::ifstream& fin, bool print)   = 0;
    virtual void   readDimacs9  (std::ifstream& fin, bool print)   = 0;
    virtual void   readDimacs10 (std::ifstream& fin, bool print)   = 0;
    virtual void   readSnap     (std::ifstream& fin, bool print)   = 0;
    virtual void   readKonect   (std::ifstream& fin, bool print)   = 0;
    virtual void   readNetRepo  (std::ifstream& fin)               = 0;
    virtual void   readBinary   (const char* filename, bool print) = 0;

    virtual GInfo  getMarketHeader   (std::ifstream& fin) final;
    virtual GInfo  getDimacs9Header  (std::ifstream& fin) final;
    virtual GInfo  getDimacs10Header (std::ifstream& fin) final;
    virtual GInfo  getKonectHeader   (std::ifstream& fin) final;
    virtual void   getNetRepoHeader  (std::ifstream& fin) final;
    virtual GInfo  getSnapHeader     (std::ifstream& fin) final;

    virtual void COOtoCSR() noexcept = 0;
    //virtual void CSRtoCOO() noexcept = 0;
};

template<typename vid_t, typename eoff_t>
inline vid_t GraphBase<vid_t, eoff_t>::nV() const noexcept {
    return _nV;
}

template<typename vid_t, typename eoff_t>
inline eoff_t GraphBase<vid_t, eoff_t>::nE() const noexcept {
    return _nE;
}

} // namespace graph
