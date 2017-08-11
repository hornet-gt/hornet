/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
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
namespace custinger {

template<unsigned SIZE>
void bind(const byte_t* (&data_ptrs)[SIZE], int start_index) noexcept {}

template<unsigned SIZE, typename T, typename... TArgs>
void bind(const byte_t* (&data_ptrs)[SIZE], int start_index,
          const T* data, TArgs... args) noexcept {
    assert(start_index < SIZE && "Index out-of-bound");
    data_ptrs[start_index] = reinterpret_cast<byte_t*>(const_cast<T*>(data));
    bind(data_ptrs, start_index++, args...);
}

//------------------------------------------------------------------------------

inline cuStingerInit::cuStingerInit(size_t num_vertices, size_t num_edges,
                                    const eoff_t* csr_offsets,
                                    const vid_t* csr_edges)
                                    noexcept :
                                           _nV(num_vertices),
                                           _nE(num_edges) {
    _vertex_data_ptrs[0] = reinterpret_cast<const byte_t*>(csr_offsets);
    _edge_data_ptrs[0]   = reinterpret_cast<const byte_t*>(csr_edges);
}

inline cuStingerInit::~cuStingerInit() noexcept {
    for (const auto& it : ptrs_to_delete)
        delete[] it;
}

inline size_t cuStingerInit::nV() const noexcept {
    return _nV;
}

inline size_t cuStingerInit::nE() const noexcept {
    return _nE;
}

inline const eoff_t* cuStingerInit::csr_offsets() const noexcept {
    return reinterpret_cast<const eoff_t*>(_vertex_data_ptrs[0]);
}

inline const vid_t* cuStingerInit::csr_edges() const noexcept {
    return reinterpret_cast<const vid_t*>(_edge_data_ptrs[0]);
}

template<typename... TArgs>
void cuStingerInit::insertVertexData(TArgs... vertex_data) noexcept {
    static_assert(sizeof...(TArgs) == NUM_EXTRA_VTYPES,
                  "Number of Vertex data type not correct");
    using T = typename xlib::tuple_rm_pointers<std::tuple<TArgs...>>::type;
    static_assert(xlib::tuple_compare<VertexTypes, T>::value,
                  "Incorrect Vertex data type");

    bind(_vertex_data_ptrs, _vertex_field_count, vertex_data...);
    _vertex_field_count += sizeof...(TArgs);
}

template<typename... TArgs>
void cuStingerInit::insertEdgeData(TArgs... edge_data) noexcept {
    static_assert(sizeof...(TArgs) == NUM_EXTRA_ETYPES,
                  "Number of Edge data type not correct");
    using T = typename xlib::tuple_rm_pointers<std::tuple<TArgs...>>::type;
    static_assert(xlib::tuple_compare<EdgeTypes, T>::value,
                  "Incorrect Edge data type");

    bind(_edge_data_ptrs, _edge_field_count, edge_data...);
    _edge_field_count += sizeof...(TArgs);
}

template<typename T>
void cuStingerInit::addVertexField() noexcept {
    auto ptr = new T[_nV]();
    _edge_data_ptrs[_vertex_field_count++] = ptr;
    ptrs_to_delete.push_back(reinterpret_cast<byte_t*>(ptr));
}

template<typename T>
void cuStingerInit::addEdgeField() noexcept {
    auto ptr = new T[_nE]();
    _edge_data_ptrs[_edge_field_count++] = ptr;
    ptrs_to_delete.push_back(reinterpret_cast<byte_t*>(ptr));
}

} // namespace custinger
