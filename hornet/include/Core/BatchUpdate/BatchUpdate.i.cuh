/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
#include "Host/Metaprogramming.hpp"

template <typename T>
void print_vec(thrust::device_vector<T>& vec) {
        thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(std::cout, " "));
        std::cout<<"\n";
}

template <typename T>
void print_vec(thrust::device_vector<T>& d, std::string name) {
  std::cout<<"\n"<<name<<" : ";
  thrust::copy(d.begin(), d.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout<<"\n";
}

namespace hornet {

#define BATCH_UPDATE_PTR BatchUpdatePtr<vid_t, TypeList<EdgeMetaTypes...>,\
                               device_t, degree_t>

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t,
    DeviceType device_t>
inline
BATCH_UPDATE_PTR::
BatchUpdatePtr(
        degree_t num_edges,
        SoAPtr<vid_t, vid_t, EdgeMetaTypes...> ptr) noexcept :
    _nE(num_edges),
    _batch_ptr(ptr) {
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t,
    DeviceType device_t>
inline
BATCH_UPDATE_PTR::
BatchUpdatePtr(
        degree_t num_edges,
        vid_t * src,
        vid_t * dst,
        EdgeMetaTypes *... edge_meta_data) noexcept :
    _nE(num_edges),
    _batch_ptr(src, dst, edge_meta_data...) {
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t,
    DeviceType device_t>
inline
BATCH_UPDATE_PTR::
BatchUpdatePtr(
    SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, device_t>& data) noexcept :
    _nE(data.get_num_items()),
    _batch_ptr(data.get_soa_ptr()) {
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t,
    DeviceType device_t>
template <unsigned N>
inline void
BATCH_UPDATE_PTR::
insertEdgeData(typename xlib::SelectType<N,
        vid_t *,
        vid_t *,
        EdgeMetaTypes *...>::type edge_data) noexcept {
    _batch_ptr.template set<N>(edge_data);
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t,
    DeviceType device_t>
inline degree_t
BATCH_UPDATE_PTR::
nE(void) const noexcept {
    return _nE;
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t,
    DeviceType device_t>
inline SoAPtr<vid_t, vid_t, EdgeMetaTypes...>
BATCH_UPDATE_PTR::
get_ptr(void) const noexcept {
    return _batch_ptr;
}

namespace gpu {

#define BATCH_UPDATE BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <DeviceType device_t>
BATCH_UPDATE::
BatchUpdate(BatchUpdatePtr<vid_t, TypeList<EdgeMetaTypes...>, device_t, degree_t> ptr) noexcept {
    reset(ptr);
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <DeviceType device_t>
BATCH_UPDATE::
BatchUpdate(hornet::COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& data) noexcept {
  BatchUpdatePtr<vid_t, TypeList<EdgeMetaTypes...>, device_t, degree_t> bPtr(data.size(), data.getPtr());
  reset(bPtr);
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <DeviceType device_t>
BATCH_UPDATE::
BatchUpdate(SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, device_t>& data) noexcept {
  BatchUpdatePtr<vid_t, TypeList<EdgeMetaTypes...>, device_t, degree_t> bPtr(data);
  reset(bPtr);
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <DeviceType device_t>
void
BATCH_UPDATE::
reset(BatchUpdatePtr<vid_t, TypeList<EdgeMetaTypes...>, device_t, degree_t> ptr) noexcept {
    _nE = ptr.nE();
    _edge[0].resize(_nE);
    _edge[1].resize(_nE);
    unique_sources.resize(_nE);
    unique_degrees.resize(_nE + 1);
    duplicate_flag.resize(_nE + 1);
    cub_runlength.resize(_nE);
    batch_offsets.resize(_nE + 1);
    graph_offsets.resize(_nE + 1);
    cub_prefixsum.resize(_nE + 1);
    cub_prefixmax.resize(_nE + 1);
    current_edge = 0;
    in_edge().copy(ptr.get_ptr(), device_t, (int)_nE);
    if (1 < sizeof...(EdgeMetaTypes)) {
        in_range().resize(_nE);
        out_range().resize(_nE);
    }

    vertex_access[0].resize(_nE);
    vertex_access[1].resize(_nE);
    host_vertex_access[0].resize(_nE);
    host_vertex_access[1].resize(_nE);
    realloc_sources.resize(_nE);
    realloc_sources_count_buffer.resize(1);
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
CSoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE>&
BATCH_UPDATE::
in_edge(void) noexcept {
    return _edge[current_edge];
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
CSoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE>&
BATCH_UPDATE::
out_edge(void) noexcept {
    return _edge[!current_edge];
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
thrust::device_vector<vid_t>&
BATCH_UPDATE::
in_range(void) noexcept {
    return range[current_edge];
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
thrust::device_vector<vid_t>&
BATCH_UPDATE::
out_range(void) noexcept {
    return range[!current_edge];
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
BATCH_UPDATE::
flip_resource(void) noexcept {
    current_edge = !current_edge;
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
BATCH_UPDATE::
sort(void) noexcept {
    auto in_ptr = in_edge().get_soa_ptr();
    auto out_ptr = out_edge().get_soa_ptr();
    //in_range().resize(_nE);
    //thrust::sequence(in_range().begin(), in_range().end());
    //TODO : Move in_range resize and sequencing to sort_batch dependent on usage
    bool flip = sort_batch(in_ptr, _nE, in_range(), out_ptr);
    if (flip) { flip_resource(); }
}

struct IsSrcDstEqual {
    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& a, const Tuple& b) {
        return ((thrust::get<0>(a) == thrust::get<0>(b)) &&
                (thrust::get<1>(a) == thrust::get<1>(b)));
    }

};

//Only src,dst in batch
template <typename degree_t, typename... EdgeTypes>
void
remove_duplicates_edges_only(
        CSoAPtr<EdgeTypes...> in_ptr,
        CSoAPtr<EdgeTypes...> out_ptr,
        degree_t& nE,
        thrust::device_vector<degree_t>& range,
        thrust::device_vector<degree_t>& range_copy) {
    auto begin_in_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                in_ptr.template get<0>(), in_ptr.template get<1>()));
    auto begin_out_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                out_ptr.template get<0>(), out_ptr.template get<1>()));
    auto end_ptr =
        thrust::unique_copy(
                thrust::device,
                begin_in_tuple, begin_in_tuple + nE,
                begin_out_tuple,
                IsSrcDstEqual());
    nE = end_ptr - begin_out_tuple;
    range.resize(nE);
    range_copy.resize(nE);
}

//Only src,dst in batch
template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 == sizeof...(EdgeTypes)), void>::type
remove_duplicates(
        CSoAPtr<EdgeTypes...> in_ptr,
        CSoAPtr<EdgeTypes...> out_ptr,
        degree_t& nE,
        thrust::device_vector<degree_t>& range,
        thrust::device_vector<degree_t>& range_copy) {
    remove_duplicates_edges_only(in_ptr, out_ptr, nE, range, range_copy);
}

//Only src,dst,meta in batch
template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(3 == sizeof...(EdgeTypes)), void>::type
remove_duplicates(
        CSoAPtr<EdgeTypes...> in_ptr,
        CSoAPtr<EdgeTypes...> out_ptr,
        degree_t& nE,
        thrust::device_vector<degree_t>& range,
        thrust::device_vector<degree_t>& range_copy) {
    auto begin_in_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                in_ptr.template get<0>(), in_ptr.template get<1>(), in_ptr.template get<2>()));
    auto begin_out_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                out_ptr.template get<0>(), out_ptr.template get<1>(), out_ptr.template get<2>()));
    auto end_ptr =
        thrust::unique_copy(
                thrust::device,
                begin_in_tuple, begin_in_tuple + nE,
                begin_out_tuple,
                IsSrcDstEqual());
    nE = end_ptr - begin_out_tuple;
    range.resize(nE);
    range_copy.resize(nE);
}

//src,dst,meta1,...,metaN in batch
template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(3 < sizeof...(EdgeTypes)), void>::type
remove_duplicates(
        CSoAPtr<EdgeTypes...> in_ptr,
        CSoAPtr<EdgeTypes...> out_ptr,
        degree_t& nE,
        thrust::device_vector<degree_t>& range,
        thrust::device_vector<degree_t>& range_copy) {
    range.resize(nE);
    range_copy.resize(nE);
    thrust::sequence(range.begin(), range.end());
    auto begin_in_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                in_ptr.template get<0>(), in_ptr.template get<1>(), range.begin()));
    auto begin_out_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                out_ptr.template get<0>(), out_ptr.template get<1>(), range_copy.begin()));
    auto end_ptr =
        thrust::unique_copy(
                thrust::device,
                begin_in_tuple, begin_in_tuple + nE,
                begin_out_tuple,
                IsSrcDstEqual());
    nE = end_ptr - begin_out_tuple;

    RecursiveGather<2, sizeof...(EdgeTypes)>::assign(in_ptr, out_ptr, range_copy, nE);
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
BATCH_UPDATE::
remove_batch_duplicates(bool insert) noexcept {
    if (_nE == 0) { return; }
    auto in_ptr = in_edge().get_soa_ptr();
    auto out_ptr = out_edge().get_soa_ptr();
    if (insert) {
        remove_duplicates(in_ptr, out_ptr, _nE, in_range(), out_range());
    } else {
        remove_duplicates_edges_only(in_ptr, out_ptr, _nE, in_range(), out_range());
    }
    flip_resource();
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
remove_graph_duplicates(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept {
    if (_nE == 0) { return; }
    _nE = in_edge().get_num_items();
    //Get unique sources and degrees
    duplicate_flag.resize(_nE + 1);
    thrust::fill(duplicate_flag.begin(), duplicate_flag.end(), 1);
    auto in_ptr = in_edge().get_soa_ptr();
    vid_t * batch_src = in_ptr.template get<0>();
    vid_t * batch_dst = in_ptr.template get<1>();

    degree_t total_work =
        get_unique_sources_meta_data(batch_src, _nE, unique_sources, unique_degrees, batch_offsets, graph_offsets, hornet_device);
    if (total_work == 0) { return; }

    //_d_degree_tmp -> graph_offsets
    //degree_tmp_sum -> graph_offsets[unique_sources_count]
    //_d_batch_offset -> batch_offsets
    //_d_unique -> unique_sources
    //num_uniques -> unique_sources_count
    mark_duplicate_edges(hornet_device,
            unique_sources,
            //batch_dst,
            in_ptr,
            batch_offsets, graph_offsets,
            duplicate_flag,
            total_work);

    cub_prefixsum.run(duplicate_flag.data().get(), _nE);

    _nE = duplicate_flag[duplicate_flag.size() - 1];
    write_unique_edges(in_edge(), out_edge(), duplicate_flag);

    flip_resource();
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
degree_t
BATCH_UPDATE::
get_unique_sources_meta_data(
        vid_t * const batch_src,
        const degree_t nE,
        thrust::device_vector<vid_t>& unique_sources,
        thrust::device_vector<degree_t>& unique_degrees,
        thrust::device_vector<degree_t>& batch_offsets,
        thrust::device_vector<degree_t>& graph_offsets,
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept {

    unique_sources.resize(_nE);
    unique_degrees.resize(_nE);
    graph_offsets.resize(_nE + 1);

    degree_t unique_sources_count = cub_runlength.run(batch_src, nE,
            unique_sources.data().get(), unique_degrees.data().get());
    batch_offsets.resize(unique_sources_count + 1);
    thrust::copy(
            unique_degrees.begin(),
            unique_degrees.begin() + unique_sources_count + 1,
            batch_offsets.begin());

    //Find offsets to the adjacency lists of the sources in the batch graph
    cub_prefixsum.run(batch_offsets.data().get(), unique_sources_count + 1);

    unique_sources.resize(unique_sources_count);
    graph_offsets.resize(unique_sources_count + 1);

    //Get degrees of batch sources in hornet graph
    get_vertex_degrees(hornet_device, unique_sources, graph_offsets);

    cub_prefixsum.run(graph_offsets.data().get(), graph_offsets.size());

    degree_t total_work = graph_offsets[graph_offsets.size() - 1];

    return total_work;
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
degree_t
BATCH_UPDATE::
get_unique_sources_meta_data_erase(
        vid_t * const batch_src,
        vid_t * const batch_dst,
        const degree_t nE,
        thrust::device_vector<vid_t>& unique_sources,
        thrust::device_vector<degree_t>& batch_src_offsets,
        thrust::device_vector<degree_t>& batch_dst_offsets,
        thrust::device_vector<degree_t>& batch_dst_degrees,
        thrust::device_vector<degree_t>& graph_offsets,
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept {

    ////DESTINATION BATCH DEGREES
    markUniqueOffsets(batch_src, batch_dst, nE, batch_dst_offsets, batch_dst_degrees, cub_prefixmax);

    ////GET BATCH SOURCES AND DEGREES
    unique_sources.resize(_nE);
    batch_src_offsets.resize(_nE);
    degree_t unique_sources_count = cub_runlength.run(batch_src, nE,
            unique_sources.data().get(), batch_src_offsets.data().get());
    unique_sources.resize(unique_sources_count);
    batch_src_offsets.resize(unique_sources_count + 1);

    ////SOURCE BATCH OFFSETS
    //Find offsets to the adjacency lists of the sources in the batch graph
    cub_prefixsum.run(batch_src_offsets.data().get(), unique_sources_count + 1);

    ////SOURCE GRAPH OFFSETS
    graph_offsets.resize(unique_sources_count + 1);
    //Get degrees of batch sources in hornet graph
    get_vertex_degrees(hornet_device, unique_sources, graph_offsets);
    cub_prefixsum.run(graph_offsets.data().get(), unique_sources_count + 1);
    degree_t total_work = graph_offsets[graph_offsets.size() - 1];

    return total_work;
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
BATCH_UPDATE::
size(void) noexcept {
    return _nE;
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
preprocess(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        bool removeBatchDuplicates,
        bool removeGraphDuplicates) noexcept {
    if (_nE == 0) { return; }
    sort();
    if (removeBatchDuplicates) {
        remove_batch_duplicates();
    }
    if (removeGraphDuplicates) {
        remove_graph_duplicates(hornet_device);
    }
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
preprocess_erase(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        bool removeBatchDuplicates) noexcept {
    if (_nE == 0) { return; }
    auto in_ptr = in_edge().get_soa_ptr();
    sort_edges(in_ptr, _nE);
    CHECK_CUDA_ERROR
    if (removeBatchDuplicates) {
        remove_batch_duplicates(false);
    CHECK_CUDA_ERROR
    }
    locateEdgesToBeErased(hornet_device, !removeBatchDuplicates);
    CHECK_CUDA_ERROR
    overWriteEdges(hornet_device);
    CHECK_CUDA_ERROR
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
overWriteEdges(hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept {
    thrust::device_vector<degree_t>& destination_edges = range[1];//reuse
    thrust::device_vector<degree_t>& destination_edges_flag = range[0];
    thrust::device_vector<degree_t>& source_edges_flag = duplicate_flag;
    thrust::device_vector<degree_t>& source_edges_offset = graph_offsets;

    auto in_ptr = in_edge().get_soa_ptr();
    vid_t * batch_src = in_ptr.template get<0>();

    thrust::device_vector<degree_t>& batch_src_degrees = unique_degrees;
    unique_sources.resize(_nE);
    batch_src_degrees.resize(_nE);
    degree_t unique_sources_count = cub_runlength.run(batch_src, _nE,
            unique_sources.data().get(), batch_src_degrees.data().get());
    unique_sources.resize(unique_sources_count);
    batch_src_degrees.resize(unique_sources_count);
    CHECK_CUDA_ERROR

    destination_edges_flag.resize(_nE);
    source_edges_flag.resize(_nE);
    source_edges_offset.resize(_nE);
    //unique_sources
    //range[1] -> erase locations
    //batch_src_degrees
    //batch_src_offsets
    markOverwriteSrcDst(hornet_device,
            batch_src,
            unique_sources,
            batch_src_degrees,
            destination_edges,
            destination_edges_flag,
            source_edges_flag,
            source_edges_offset
            );
    CHECK_CUDA_ERROR

}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
markOverwriteSrcDst(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        vid_t * batch_src,
        thrust::device_vector<vid_t> &unique_sources,
        thrust::device_vector<degree_t>& batch_src_degrees,
        thrust::device_vector<degree_t>& destination_edges,
        thrust::device_vector<degree_t>& destination_edges_flag,
        thrust::device_vector<degree_t>& source_edges_flag,
        thrust::device_vector<degree_t>& source_edges_offset
        ) noexcept {

    thrust::device_vector<degree_t>& batch_src_offsets = batch_offsets;
    batch_src_offsets.resize(batch_src_degrees.size() + 1);
    thrust::copy(batch_src_degrees.begin(), batch_src_degrees.end(), batch_src_offsets.begin());
    cub_prefixsum.run(batch_src_offsets.data().get(), batch_src_degrees.size() + 1);
    thrust::fill(source_edges_flag.begin(), source_edges_flag.end(), 1);
    thrust::fill(destination_edges_flag.begin(), destination_edges_flag.end(), 0);
    CHECK_CUDA_ERROR

    degree_t total_work = batch_src_offsets[batch_src_offsets.size() - 1];
    const int BLOCK_SIZE = 256;
    int smem = xlib::DeviceProperty::smem_per_block<degree_t>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(total_work, smem);
    if (num_blocks != 0) {
    markOverwriteSrcDstKernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE>>>(
                hornet_device,
                unique_sources.data().get(),
                batch_src_offsets.data().get(),
                batch_src_degrees.data().get(),
                destination_edges.data().get(),
                destination_edges_flag.data().get(),//new
                source_edges_flag.data().get(),//new
                source_edges_offset.data().get(),//new
                batch_src_offsets.size()
                );
    }
    CHECK_CUDA_ERROR

    thrust::device_ptr<vid_t> sources(batch_src);
    realloc_sources.resize(destination_edges.size());
    batch_src_offsets.resize(destination_edges.size());
    auto ptr_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                sources, destination_edges.begin()));
    auto out_ptr_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                realloc_sources.begin(), batch_src_offsets.begin()));
    degree_t length =
    thrust::copy_if(ptr_tuple, ptr_tuple + destination_edges.size(),
            destination_edges_flag.begin(),
            out_ptr_tuple, thrust::identity<degree_t>()) - out_ptr_tuple;
    if (length == 0) { return; }
    realloc_sources.resize(length);
    batch_src_offsets.resize(length);
    CHECK_CUDA_ERROR

    destination_edges.resize(realloc_sources.size());
    length = thrust::copy_if(source_edges_offset.begin(), source_edges_offset.end(),
            source_edges_flag.begin(),
            destination_edges.begin(), thrust::identity<degree_t>()) - destination_edges.begin();
    destination_edges.resize(length);
    overwriteDeletedEdges(hornet_device, realloc_sources, batch_src_offsets, destination_edges);
    CHECK_CUDA_ERROR
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
locateEdgesToBeErased(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device, bool duplicate_edges_present) noexcept {
    _nE = in_edge().get_num_items();
    if (_nE == 0) { return; }
    auto in_ptr = in_edge().get_soa_ptr();
    vid_t * batch_src = in_ptr.template get<0>();
    vid_t * batch_dst = in_ptr.template get<1>();
    thrust::device_vector<degree_t>& batch_src_offsets = batch_offsets;
    thrust::device_vector<degree_t>& batch_dst_offsets = unique_degrees;
    thrust::device_vector<degree_t>& batch_dst_degrees  = duplicate_flag;

    degree_t total_work =
        get_unique_sources_meta_data_erase(
                batch_src, batch_dst, _nE,
                unique_sources,
                batch_src_offsets,
                batch_dst_offsets,
                batch_dst_degrees,
                graph_offsets,
                hornet_device);
    if (total_work == 0) { return; }

    thrust::device_vector<degree_t>& erase_edge_location = range[0];//reuse
    thrust::device_vector<degree_t>& batch_erase_flag = unique_degrees;//reuse
    locate_erased_edges(hornet_device,
            unique_sources,
            batch_dst,
            batch_src_offsets,
            batch_dst_degrees,
            graph_offsets,
            batch_erase_flag,
            erase_edge_location,
            total_work);

    out_edge().resize(_nE);
    auto out_ptr = out_edge().get_soa_ptr();
    vid_t * batch_src_out = out_ptr.template get<0>();

    thrust::device_vector<degree_t>& destination_edges = range[1];//reuse
    //realloc_sources.resize(_nE);
    destination_edges.resize(_nE);
    auto ptr_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                batch_src, erase_edge_location.begin()));
    auto out_ptr_tuple = thrust::make_zip_iterator(thrust::make_tuple(
                batch_src_out, destination_edges.begin()));
                //realloc_sources.begin(), destination_edges.begin()));
    _nE = thrust::copy_if(thrust::device,
            ptr_tuple, ptr_tuple + _nE,
            batch_erase_flag.begin(),
            out_ptr_tuple,
            thrust::identity<degree_t>()) - out_ptr_tuple;
    out_edge().resize(_nE);
    //realloc_sources.resize(length);
    destination_edges.resize(_nE);
    flip_resource();
    //in_edge now contains all the edges that are present in hornet and are
    //meant to be removed
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
get_reallocate_vertices_meta_data(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_realloc_v_data,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_new_v_data,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_realloc_v_data,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_new_v_data,
        degree_t& reallocated_vertices_count,
        const bool is_insert) {

    vertex_access[0].resize(_nE);
    vertex_access[1].resize(_nE);
    host_vertex_access[0].resize(_nE);
    host_vertex_access[1].resize(_nE);

    vid_t * batch_src = in_edge().get_soa_ptr().template get<0>();
    unique_sources.resize(_nE);
    unique_degrees.resize(_nE + 1);

    //TODO : see if calculating run length is avoidable since batch delete does this anyway
    degree_t unique_sources_count = cub_runlength.run(batch_src, _nE,
            unique_sources.data().get(), unique_degrees.data().get());
    if (unique_sources_count == 0) { return; }
    realloc_sources.resize(unique_sources_count);
    unique_degrees.resize(unique_sources_count + 1);

    if (is_insert) {
        graph_offsets.resize(unique_sources_count + 1);//contains degrees of batch vertices in graph
    }

    d_realloc_v_data = vertex_access[0].get_soa_ptr();
    d_new_v_data = vertex_access[1].get_soa_ptr();
    h_realloc_v_data = host_vertex_access[0].get_soa_ptr();
    h_new_v_data = host_vertex_access[1].get_soa_ptr();

    degree_t * old_degree = graph_offsets.data().get();
    const int BLOCK_SIZE = 128;
    buildReallocateVerticesQueue
        <<< xlib::ceil_div<BLOCK_SIZE>(unique_sources_count), BLOCK_SIZE >>>
        (hornet_device,
         unique_sources.data().get(), unique_degrees.data().get(), unique_sources_count,
         realloc_sources.data().get(),
         d_realloc_v_data,
         d_new_v_data,
         realloc_sources_count_buffer.data().get(),
         is_insert,
         is_insert ? old_degree : nullptr
        );
    reallocated_vertices_count = realloc_sources_count_buffer[0];
    RecursiveCopy<0, 3>::copy(
            d_realloc_v_data, DeviceType::DEVICE,
            h_realloc_v_data, DeviceType::HOST,
            reallocated_vertices_count);
    DeviceCopy::copy(
            d_new_v_data. template get<0>(), DeviceType::DEVICE,
            h_new_v_data. template get<0>(), DeviceType::HOST,
            reallocated_vertices_count);
    realloc_sources.resize(reallocated_vertices_count);
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
move_adjacency_lists(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...> vertex_access_ptr,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_realloc_v_data,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_new_v_data,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_realloc_v_data,
        SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_new_v_data,
        const degree_t reallocated_vertices_count,
        const bool is_insert) {
    //Get reallocated block data to device
    RecursiveCopy<1, 3>::copy(
            h_new_v_data, DeviceType::HOST,
            d_new_v_data, DeviceType::DEVICE,
            reallocated_vertices_count);
    CUDA_CHECK_LAST()

    //Get offsets for binarySearchLB kernel
    duplicate_flag.resize(reallocated_vertices_count + 1);
    //DeviceCopy::copy(
    //        d_realloc_v_data. template get<0>(), DeviceType::DEVICE,
    //        duplicate_flag.data().get(), DeviceType::DEVICE,
    //        reallocated_vertices_count);
    CUDA_CHECK_LAST()
    if (is_insert) {
    cub_prefixsum.run(d_realloc_v_data. template get<0>(),
            duplicate_flag.size(),
            duplicate_flag.data().get());
    } else {
    cub_prefixsum.run(d_new_v_data. template get<0>(),
            duplicate_flag.size(),
            duplicate_flag.data().get());
    }
    CUDA_CHECK_LAST()
    degree_t total_work = duplicate_flag[duplicate_flag.size() - 1];
    if (total_work != 0)  {
      const int BLOCK_SIZE = 256;
      int smem = xlib::DeviceProperty::smem_per_block<degree_t>(BLOCK_SIZE);
      int num_blocks = xlib::ceil_div(total_work, smem);
      move_adjacency_lists_kernel<BLOCK_SIZE>
          <<<num_blocks, BLOCK_SIZE>>>(
                  hornet_device,
                  d_realloc_v_data,
                  d_new_v_data,
                  duplicate_flag.data().get(),
                  duplicate_flag.size());
      CUDA_CHECK_LAST()
    }
    if (reallocated_vertices_count != 0) {
      const int BLOCK_SIZE = 256;
      int num_blocks = xlib::ceil_div(reallocated_vertices_count, BLOCK_SIZE);
      set_vertex_meta_data
          <<<num_blocks, BLOCK_SIZE>>>(
                  realloc_sources.data().get(),
                  vertex_access_ptr,
                  d_new_v_data,
                  reallocated_vertices_count);
      CUDA_CHECK_LAST()
    }
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
template <typename... VertexMetaTypes>
void
BATCH_UPDATE::
appendBatchEdges(hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept {
    if (_nE == 0) { return; }
    cub_prefixsum.run(unique_degrees.data().get(), unique_degrees.size());
    degree_t total_work = unique_degrees[unique_degrees.size() - 1];

    degree_t * old_degree = graph_offsets.data().get();
    const int BLOCK_SIZE = 256;
    int smem = xlib::DeviceProperty::smem_per_block<degree_t>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(total_work, smem);
    appendBatchEdgesKernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE>>>(
                hornet_device,
                unique_sources.data().get(),
                unique_degrees.data().get(),
                old_degree,
                unique_degrees.size(),
                in_edge().get_soa_ptr().get_tail());
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
BATCH_UPDATE::
print(bool sort) noexcept {
    auto ptr = in_edge().get_soa_ptr();
    std::cout<<"src, dst : "<<size()<<"\n";
    thrust::device_vector<vid_t> src(size());
    thrust::device_vector<vid_t> dst(size());
    thrust::copy(ptr.template get<0>(), ptr.template get<0>() + size(), src.begin());
    thrust::copy(ptr.template get<1>(), ptr.template get<1>() + size(), dst.begin());
    if (!sort) {
      thrust::copy(src.begin(), src.end(), std::ostream_iterator<vid_t>(std::cout, " "));
      std::cout<<"\n";
      thrust::copy(dst.begin(), dst.end(), std::ostream_iterator<vid_t>(std::cout, " "));
      std::cout<<"\n";
    } else {
      thrust::host_vector<vid_t> hSrc = src;
      thrust::host_vector<vid_t> hDst = dst;
      std::vector<std::pair<vid_t, vid_t>> e;
      for (int i = 0; i < size(); ++i) {
        e.push_back(std::make_pair(hSrc[i], hDst[i]));
      }
      std::sort(e.begin(), e.end());
      for (unsigned i = 0; i < e.size(); ++i) { std::cout<<e[i].first<<" "; }
      std::cout<<"\n";
      for (unsigned i = 0; i < e.size(); ++i) { std::cout<<e[i].second<<" "; }
      std::cout<<"\n";
    }
}

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
BATCH_UPDATE::
nE(void) const noexcept {
    return _nE;
}

}//namespace gpu
}//namespace hornet
