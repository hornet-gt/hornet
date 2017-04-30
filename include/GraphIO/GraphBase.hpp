/**
 * @internal
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

#include <string>   //std::string

namespace graph {

class Property {
    template<typename, typename> friend class GraphBase;
    template<typename, typename> friend class GraphStd;
    template<typename, typename, typename> friend class GraphWeight;
public:
    enum Enum { PRINT = 1, RANDOMIZE = 2, SORT = 4 };
    explicit Property(int state = 0) noexcept;
private:
    int _state;

    bool is_undefined() const noexcept;
    bool is_sort()      const noexcept;
    bool is_randomize() const noexcept;
    bool is_print()     const noexcept;
};

class Structure {
    template<typename, typename> friend class GraphBase;
    template<typename, typename> friend class GraphStd;
    template<typename, typename, typename> friend class GraphWeight;
public:
    enum Enum { DIRECTED = 1, UNDIRECTED = 2, REVERSE = 4, COO = 8 };
    explicit Structure(int state = 0) noexcept;
    void operator|=(int value)        noexcept;
private:
    enum WType { NONE, INTEGER, REAL };
    int   _state;
    WType _wtype { NONE };

    bool is_undefined()     const noexcept;
    bool is_directed()      const noexcept;
    bool is_undirected()    const noexcept;
    bool is_reverse()       const noexcept;
    bool is_coo()           const noexcept;
    bool is_direction_set() const noexcept;
    bool is_weighted()      const noexcept;
};


template<typename vid_t, typename eoff_t>
class GraphBase {
public:
    virtual vid_t  nV() const noexcept final;
    virtual eoff_t nE() const noexcept final;
    virtual const std::string& name() const noexcept final;

    virtual void read(const char* filename,                             //NOLINT
                      Property prop = Property(Property::PRINT)) final;
    virtual void print()     const noexcept = 0;
    virtual void print_raw() const noexcept = 0;

    GraphBase(const GraphBase&)      = delete;
    void operator=(const GraphBase&) = delete;
protected:
    std::string _graph_name { "" };
    vid_t        _nV { 0 };
    eoff_t       _nE { 0 };
    Structure   _structure;

    explicit GraphBase(Structure structure = Structure(Structure::REVERSE))
                       noexcept;
    virtual ~GraphBase() noexcept = default;
    virtual void   allocate() noexcept = 0;

    virtual void   readMarket   (std::ifstream& fin, Property property)   = 0;
    virtual void   readDimacs9  (std::ifstream& fin, Property property)   = 0;
    virtual void   readDimacs10 (std::ifstream& fin, Property property)   = 0;
    virtual void   readSnap     (std::ifstream& fin, Property property)   = 0;
    virtual void   readKonect   (std::ifstream& fin, Property property)   = 0;
    virtual void   readNetRepo  (std::ifstream& fin, Property property)   = 0;
    virtual void   readBinary   (const char* filename, Property property) = 0;

    virtual size_t getMarketHeader   (std::ifstream& fin) final;
    virtual size_t getDimacs9Header  (std::ifstream& fin) final;
    virtual void   getDimacs10Header (std::ifstream& fin) final;
    virtual void   getKonectHeader   (std::ifstream& fin) final;
    virtual void   getNetRepoHeader  (std::ifstream& fin) final;
    virtual size_t getSnapHeader     (std::ifstream& fin) final;

    virtual void  print_property() final;
};

} // namespace graph

#include "GraphBase.i.hpp"
