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
#ifndef BATCHUPDATE_CUH
#define BATCHUPDATE_CUH
#include "../Conf/HornetConf.cuh"
#include "../Conf/Common.cuh"
#include "../HornetDevice/HornetDevice.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <Device/Primitives/CubWrapper.cuh>
#include <Device/Primitives/BinarySearchLB.cuh>
#include <Device/Util/DeviceQueue.cuh>
#include "BatchUpdateKernels.cuh"
#include "../Static/Static.cuh"

#define CUDA_TRY( call ) 									                            \
{                                                                     \
    cudaError_t cudaStatus = call;                                    \
    if ( cudaSuccess != cudaStatus )                                  \
    {                                                                 \
        std::cerr << "ERROR: CUDA Runtime call " << #call             \
                  << " in line " << __LINE__                            \
                  << " of file " << __FILE__                            \
                  << " failed with " << cudaGetErrorString(cudaStatus)  \
                  << " (" << cudaStatus << ").\n";                     \
    }												                                          \
}

#define CUDA_CHECK_LAST() CUDA_TRY(cudaPeekAtLastError())


namespace hornet {

template <
         typename, typename = EMPTY,
         DeviceType = DeviceType::DEVICE,
         typename = int>
         class BatchUpdatePtr;

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t,
    DeviceType device_t>
class BatchUpdatePtr<
    vid_t,
    TypeList<EdgeMetaTypes...>,
    device_t, degree_t> {

    degree_t                       _nE        { 0 };

    SoAPtr<vid_t, vid_t, EdgeMetaTypes...>  _batch_ptr;

public:

    BatchUpdatePtr(
            degree_t num_edges,
            SoAPtr<vid_t, vid_t, EdgeMetaTypes...> ptr) noexcept;

    BatchUpdatePtr(
            degree_t num_edges,
            vid_t*  src,
            vid_t*  dst,
            EdgeMetaTypes *... edge_data) noexcept;

    BatchUpdatePtr(SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, device_t>& data) noexcept;

    template <unsigned N>
    void insertEdgeData(typename xlib::SelectType<N,
            vid_t *,
            vid_t *,
            EdgeMetaTypes *...>::type edge_data) noexcept;

    degree_t nE(void) const noexcept;

    SoAPtr<vid_t, vid_t, EdgeMetaTypes...> get_ptr(void) const noexcept;

};

namespace gpu {

template <typename, typename = EMPTY, typename = int> class BatchUpdate;

template <typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
class BatchUpdate<
    vid_t, TypeList<EdgeMetaTypes...>, degree_t> {

    degree_t                       _nE        { 0 };

    CSoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE> _edge[2];

    bool current_edge;

    thrust::device_vector<degree_t> range[2];

    thrust::device_vector<vid_t>    unique_sources;
    thrust::device_vector<degree_t> unique_degrees;
    thrust::device_vector<degree_t> duplicate_flag;
    thrust::device_vector<degree_t>  batch_offsets;
    thrust::device_vector<degree_t>  graph_offsets;

    xlib::CubRunLengthEncode<vid_t>  cub_runlength;
    xlib::CubExclusiveSum<degree_t>  cub_prefixsum;
    xlib::CubInclusiveMax<degree_t>  cub_prefixmax;

    thrust::device_vector<vid_t>   realloc_sources;

    SoAData<TypeList<degree_t, xlib::byte_t*, degree_t, degree_t>, DeviceType::DEVICE> vertex_access[2];
    SoAData<TypeList<degree_t, xlib::byte_t*, degree_t, degree_t>, DeviceType::HOST> host_vertex_access[2];

    thrust::device_vector<degree_t> realloc_sources_count_buffer;

    //Functions

    thrust::device_vector<vid_t>& in_range() noexcept;

    thrust::device_vector<vid_t>& out_range() noexcept;

    void flip_resource(void) noexcept;

    template <typename... VertexMetaTypes>
    void remove_graph_duplicates(
            hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept;

    template <typename... VertexMetaTypes>
    degree_t get_unique_sources_meta_data(
            vid_t * const batch_src,
            const degree_t nE,
            thrust::device_vector<vid_t>& unique_sources,
            thrust::device_vector<degree_t>& unique_degrees,
            thrust::device_vector<degree_t>& batch_offsets,
            thrust::device_vector<degree_t>& graph_offsets,
            hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept;

    template <typename... VertexMetaTypes>
    degree_t get_unique_sources_meta_data_erase(
        vid_t * const batch_src,
        vid_t * const batch_dst,
        const degree_t nE,
        thrust::device_vector<vid_t>& unique_sources,
        thrust::device_vector<degree_t>& batch_src_offsets,
        thrust::device_vector<degree_t>& batch_dst_offsets,
        thrust::device_vector<degree_t>& batch_dst_degrees,
        thrust::device_vector<degree_t>& graph_offsets,
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept;

    template <typename... VertexMetaTypes>
    void overWriteEdges(hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept;

    public :

    using VertexAccessT = SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>;

    template <DeviceType device_t>
    BatchUpdate(BatchUpdatePtr<vid_t, TypeList<EdgeMetaTypes...>, device_t, degree_t> ptr) noexcept;

    template <DeviceType device_t>
    BatchUpdate(SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, device_t>& data) noexcept;

    template <DeviceType device_t>
    BatchUpdate(hornet::COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& data) noexcept;

    template <DeviceType device_t>
    void reset(BatchUpdatePtr<vid_t, TypeList<EdgeMetaTypes...>, device_t, degree_t> ptr) noexcept;

    void sort(void) noexcept;

    template <typename... VertexMetaTypes>
    void preprocess(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        bool removeBatchDuplicates,
        bool removeGraphDuplicates) noexcept;

    void remove_batch_duplicates(bool insert = true) noexcept;

    CSoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE>&
    in_edge(void) noexcept;

    CSoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE>&
    out_edge(void) noexcept;

    degree_t size(void) noexcept;

    template <typename... VertexMetaTypes>
    void
    get_reallocate_vertices_meta_data(
            hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_realloc_v_data,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_new_v_data,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_realloc_v_data,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_new_v_data,
            degree_t& reallocated_vertices_count,
            const bool is_insert);

    template <typename... VertexMetaTypes>
    void
    move_adjacency_lists(
            hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...> vertex_access_ptr,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_realloc_v_data,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& h_new_v_data,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_realloc_v_data,
            SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t>& d_new_v_data,
            const degree_t reallocated_vertices_count,
            const bool is_insert);

    template <typename... VertexMetaTypes>
    void
    appendBatchEdges(
            hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device) noexcept;

    void print(bool sort = false) noexcept;

    template <typename... VertexMetaTypes>
    void preprocess_erase(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        bool removeBatchDuplicates) noexcept;

    template <typename... VertexMetaTypes>
    void locateEdgesToBeErased(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        bool duplicate_edges_present) noexcept;

    template <typename... VertexMetaTypes>
    void markOverwriteSrcDst(
        hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>& hornet_device,
        vid_t * batch_src,
        thrust::device_vector<vid_t> &unique_sources,
        thrust::device_vector<degree_t>& batch_src_degrees,
        thrust::device_vector<degree_t>& destination_edges,
        thrust::device_vector<degree_t>& destination_edges_flag,
        thrust::device_vector<degree_t>& source_edge_flag,
        thrust::device_vector<degree_t>& source_edge_offset) noexcept;

    degree_t nE(void) const noexcept;
};

}

}

#include "BatchUpdate.i.cuh"
#endif
