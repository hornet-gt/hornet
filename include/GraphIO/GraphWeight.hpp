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
#include <tuple>  //std::tuple

namespace graph {

template<typename id_t, typename off_t>
class BFS;

template<typename id_t = int, typename off_t = int, typename weight_t = int>
class GraphWeight : public GraphBase<id_t, off_t> {
    using    coo_t = typename std::tuple<id_t, id_t, weight_t>;
    using degree_t = int;
    friend class BFS<id_t, off_t>;

public:
    explicit GraphWeight()                                   noexcept = default;
    explicit GraphWeight(Structure Structure)                noexcept;
    explicit GraphWeight(const char* filename, Property prop) noexcept;
    explicit GraphWeight(Structure Structure, const char* filename,
                         Property property) noexcept;
    virtual ~GraphWeight() noexcept final;                              //NOLINT

    degree_t out_degree(id_t index)  const noexcept;
    degree_t in_degree (id_t index)  const noexcept;

    const coo_t*    coo_array()         const noexcept;
    const off_t*    out_offsets_array() const noexcept;
    const off_t*    in_offsets_array()  const noexcept;
    const id_t*     out_edges_array()   const noexcept;
    const id_t*     in_edges_array()    const noexcept;
    const degree_t* out_degrees_array() const noexcept;
    const degree_t* in_degrees_array()  const noexcept;
    const weight_t* out_weights_array() const noexcept;
    const weight_t* in_weights_array()  const noexcept;

    void print()     const noexcept override;
    void print_raw() const noexcept override;
    void toBinary(const std::string& filename, bool print = true) const;
    void toMarket(const std::string& filename) const;
private:
    off_t     *_out_offsets { nullptr };
    off_t     *_in_offsets  { nullptr };
    id_t      *_out_edges   { nullptr };
    id_t      *_in_edges    { nullptr };
    degree_t* _out_degrees  { nullptr };
    degree_t* _in_degrees   { nullptr };
    coo_t*    _coo_edges    { nullptr };
    weight_t* _out_weights  { nullptr };
    weight_t* _in_weights   { nullptr };
    size_t    _coo_size     { 0 };
    using GraphBase<id_t, off_t>::_nE;
    using GraphBase<id_t, off_t>::_nV;
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
};

} // namespace graph

#include "GraphWeight.i.hpp"
