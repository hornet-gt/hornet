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

#include <string>

namespace graph {

class Property {
    template<typename id_t, typename off_t> friend class GraphBase;
    template<typename id_t, typename off_t> friend class GraphStd;
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
    template<typename id_t, typename off_t> friend class GraphBase;
    template<typename id_t, typename off_t> friend class GraphStd;
public:
    enum Enum { DIRECTED = 1, UNDIRECTED = 2, REVERSE = 4, COO = 8 };
    explicit Structure(int state = 0) noexcept;
    void operator|=(int value)        noexcept;
private:
    int _state;

    bool is_undefined()     const noexcept;
    bool is_directed()      const noexcept;
    bool is_undirected()    const noexcept;
    bool is_reverse()       const noexcept;
    bool is_coo()           const noexcept;
    bool is_direction_set() const noexcept;
};


template<typename id_t, typename off_t>
class GraphBase {
public:
    virtual id_t  nV() const noexcept final;
    virtual off_t nE() const noexcept final;
    virtual const std::string& name() const noexcept final;

    virtual void read(const char* filename,                             //NOLINT
                      Property prop = Property(Property::PRINT)) final;
    virtual void print()     const noexcept = 0;
    virtual void print_raw() const noexcept = 0;

    GraphBase(const GraphBase&)      = delete;
    void operator=(const GraphBase&) = delete;
protected:
    std::string _graph_name { "" };
    id_t        _V { 0 };
    off_t       _E { 0 };
    Structure   _structure;

    explicit GraphBase(Structure structure = Structure(Structure::REVERSE))
                       noexcept;
    virtual ~GraphBase() noexcept;
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
