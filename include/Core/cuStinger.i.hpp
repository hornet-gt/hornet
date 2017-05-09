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

template<unsigned INDEX = 0, unsigned SIZE, typename T, typename... TArgs>
void bind(byte_t* (&data_ptrs)[SIZE], const T* data, TArgs... args) noexcept;

//==============================================================================
///////////////////
// cuStingerInit //
///////////////////

template<typename... TArgs>
void cuStingerInit::insertVertexData(TArgs... vertex_data) noexcept {
    static_assert(sizeof...(TArgs) == NUM_EXTRA_VTYPES,
                  "Number of Vertex data type not correct");
    using T = typename xlib::tuple_rm_pointers<std::tuple<TArgs...>>::type;
    static_assert(xlib::tuple_compare<VertexTypes, T>::value,
                  "Incorrect Vertex data type");

    bind<1>(_vertex_data_ptrs, vertex_data...);
}

template<typename... TArgs>
void cuStingerInit::insertEdgeData(TArgs... edge_data) noexcept {
    static_assert(sizeof...(TArgs) == NUM_EXTRA_ETYPES,
                  "Number of Edge data type not correct");
    using T = typename xlib::tuple_rm_pointers<std::tuple<TArgs...>>::type;
    static_assert(xlib::tuple_compare<EdgeTypes, T>::value,
                  "Incorrect Edge data type");

    bind<1>(_edge_data_ptrs, edge_data...);
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

//==============================================================================
///////////////
// cuStinger //
///////////////

inline size_t cuStinger::nV() const noexcept {
    return _nV;
}

inline size_t cuStinger::nE() const noexcept {
    return _nE;
}

// TO CHECK !!!!
inline const eoff_t* cuStinger::csr_offsets() const noexcept {
    return _custinger_init.csr_offsets();
}

// TO CHECK !!!!
inline const vid_t* cuStinger::csr_edges() const noexcept {
    return _custinger_init.csr_edges();
}

inline const eoff_t* cuStinger::device_csr_offsets() noexcept {
    if (_d_csr_offsets != nullptr) {
        cuMalloc(_d_csr_offsets, _nV + 1);
        cuMemcpyToDevice(csr_offsets(), _nV + 1, _d_csr_offsets);
    }
    return _d_csr_offsets;
}

inline cuStingerDevData cuStinger::device_data() const noexcept {
    cuStingerDevData data;
    data.nV = _nV;
    std::copy(_d_vertex_ptrs, _d_vertex_ptrs + NUM_VTYPES, data.d_vertex_ptrs);
    return data;
}

//==============================================================================
/////////////////
// BatchUpdate //
/////////////////

inline BatchProperty::BatchProperty(bool sort, bool weighted_distr, bool print)
                                    noexcept:
                                _sort(sort),
                                _weighted_distr(weighted_distr),
                                _print(print) {}

inline BatchUpdate::BatchUpdate(size_t batch_size) noexcept:
                                _batch_size(batch_size) {}

template<typename... TArgs>
void BatchUpdate::insertEdgeData(TArgs... edge_data) noexcept {
    static_assert(sizeof...(TArgs) + 2 == NUM_EXTRA_ETYPES,
                  "Number of Edge data type not correct");
    using T = typename xlib::tuple_rm_pointers<std::tuple<TArgs...>>::type;
    using R = typename xlib::TupleConcat<std::tuple<vid_t, vid_t>,
                                         EdgeTypes>::type;
    static_assert(xlib::tuple_compare<R, T>::value, "Incorrect Edge data type");

    bind(_edge_data_ptrs, edge_data...);
}

//==============================================================================

template<unsigned INDEX, unsigned SIZE>
void bind(byte_t* (&data_ptrs)[SIZE]) noexcept {}

template<unsigned INDEX, unsigned SIZE, typename T, typename... TArgs>
void bind(byte_t* (&data_ptrs)[SIZE], const T* data, TArgs... args) noexcept {
    static_assert(INDEX < SIZE, "Index out-of-bound");
    data_ptrs[INDEX] = reinterpret_cast<byte_t*>(const_cast<T*>(data));
    bind<INDEX + 1>(data_ptrs, args...);
}

//==============================================================================



/*
template<unsigned INDEX, typename T, typename... TArgs>
void cuStinger::insertEdgeBatch(const T* edge_data, TArgs... args) {
    static edge_t* d_batch_ptr = nullptr;
    if (INDEX == 0) {
        cuMalloc(d_batch_ptr, size);
    }
    cuMemcpyToDevice(edge_data, size, reinterpret_cast<T*>(d_batch_ptr));
    insertEdgeBatch(TArgs...);

    if (INDEX == NUM_ETYPES) {

    }
}

template<typename EqualOp>
__global__ void insertBatchKernel(edge_t* batch_ptr,
                            EqualOp equal_op = [](const Edge& a, const Edge& b){
                                                    return false;
                                                }) {

    int     id = blockIdx.x * BLOCK_DIM + gridDim.x;
    int stride = blockIdx.x * BLOCK_DIM + gridDim.x;

    for (int i = id; i < batch_size; i++) {
        auto     src_id = reinterpret_cast<vid_t*>(batch_ptr)[i];
        auto batch_edge = Edge(batch_ptr + batch_size, batch_size);
        auto        dst = batch_edge.dst();

        auto batch_vertex = Vertex(batch_src);
        auto   degree_ptr = batch_vertex.degree_ptr();
        auto       degree = batch_vertex.degree();
        auto     adj_list = batch_vertex.adj_list();

        for (int j = 0; j < degree; j++) {
            if (equal_op(batch_edge, batch_vertex.edge(j))
                break;
        }
        degree_t old_pos = atomicAdd(degree_ptr, 1);
        batch_vertex.store(batch_edge, old_pos);
    }
}

template<unsigned INDEX>
void cuStinger::insertEdgeBatch() {
    Timer<DEVICE> TM;
    TM.start();

    insertBatchKernel <<< xlib:ceil_div<BLOCK_DIM>(size), BLOCK_DIM >>>
        (d_batch_ptr);

    TM.stop();
    TM.print("insertBatchKernel");
    CHECK_CUDA_ERROR
    cuFree(d_batch_ptr);
}*/

} // namespace custinger
