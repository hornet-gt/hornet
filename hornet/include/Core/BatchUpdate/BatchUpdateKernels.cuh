#include "../Conf/EdgeOperations.cuh"

template <typename HornetDeviceT, typename vid_t, typename degree_t>
__global__
void get_vertex_degrees_kernel(
        HornetDeviceT hornet,
        const vid_t * __restrict__ vertex_id,
        const size_t vertex_id_count,
        degree_t *    __restrict__ vertex_degrees) {
    size_t     id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (auto i = id; i < vertex_id_count; i += stride)
        vertex_degrees[i] = hornet.vertex(vertex_id[i]).degree();
}

template <typename HornetDeviceT, typename vid_t, typename degree_t>
void get_vertex_degrees(HornetDeviceT& hornet,
        thrust::device_vector<vid_t>& vertex_ids,
        thrust::device_vector<degree_t>& vertex_degrees) {
    const unsigned BLOCK_SIZE = 128;
    get_vertex_degrees_kernel
        <<< xlib::ceil_div<BLOCK_SIZE>(vertex_ids.size()), BLOCK_SIZE >>>
        (hornet, vertex_ids.data().get(), vertex_ids.size(), vertex_degrees.data().get());
}

template <int BLOCK_SIZE, typename HornetDeviceT, typename vid_t, typename degree_t, typename SoAPtrT>
__global__
void mark_duplicate_edges_kernel(
        HornetDeviceT hornet,//
        const size_t graph_offsets_count,//
        const degree_t * __restrict__ graph_offsets,//
        const degree_t * __restrict__ batch_offsets,//
        const vid_t    * __restrict__ unique_src_ids,//
        SoAPtrT                          batch_edges,
        degree_t * __restrict__ duplicate_flag) {
    const vid_t * batch_dst_ids = batch_edges.template get<1>();

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&] (int pos, degree_t offset) {
                    auto     vertex = hornet.vertex(unique_src_ids[pos]);
                    assert(offset < vertex.degree());
                    auto e = vertex.edge(offset);
                    auto        dst = e.dst_id();
                    int start = batch_offsets[pos];
                    int end   = batch_offsets[pos + 1];
                    int found = xlib::lower_bound_left(
                            batch_dst_ids + start,
                            end - start,
                            dst);
                    if ((found >= 0) && (dst == batch_dst_ids[start + found])) {
                        duplicate_flag[start + found] = 0;
                    }
                };
    xlib::binarySearchLB<BLOCK_SIZE>(graph_offsets, graph_offsets_count, smem, lambda);
}

template <int BLOCK_SIZE, typename HornetDeviceT, typename vid_t, typename degree_t>
__global__
void markOverwriteSrcDstKernel(
        HornetDeviceT hornet,//
        const vid_t    * __restrict__ unique_src_ids,//
        const degree_t * __restrict__ batch_src_offsets,//
        const degree_t * __restrict__ batch_src_degrees,//
        const degree_t * __restrict__ erase_locations,//
        degree_t * __restrict__ destination_edges_flag,//
        degree_t * __restrict__ source_edges_flag,//
        degree_t * __restrict__ source_edges_offset,//
        const size_t batch_src_offsets_count) {

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&] (int pos, degree_t offset) {
        auto     vertex = hornet.vertex(unique_src_ids[pos]);
        auto batch_degree = batch_src_degrees[pos];
        assert(offset < batch_degree);
        int start = batch_src_offsets[pos];
        degree_t erase_cutoff = vertex.degree() - batch_degree;
        degree_t erase_loc = erase_locations[start + offset];
        if (erase_loc < erase_cutoff) {
            destination_edges_flag[start + offset] = 1;
        } else {
            source_edges_flag[start + erase_loc - erase_cutoff] = 0;
        }
        source_edges_offset[start + offset] = erase_cutoff + offset;
    };

    xlib::binarySearchLB<BLOCK_SIZE>(batch_src_offsets, batch_src_offsets_count, smem, lambda);
}

//Sets false to all locations in duplicate_flag if the corresponding batch_dst_ids
//is present in the graph
template <typename HornetDeviceT, typename vid_t, typename degree_t, typename SoAPtrT>
void mark_duplicate_edges(
        HornetDeviceT& hornet,
        thrust::device_vector<vid_t>& vertex_ids,
        //const vid_t * batch_dst_ids,
        SoAPtrT batch_edges,
        thrust::device_vector<degree_t>& batch_offsets,
        thrust::device_vector<degree_t>& graph_offsets,
        thrust::device_vector<degree_t>& duplicate_flag,
        const degree_t total_work) {
    const unsigned BLOCK_SIZE = 128;
    int smem = xlib::DeviceProperty::smem_per_block<degree_t>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(total_work, smem);
    mark_duplicate_edges_kernel<BLOCK_SIZE>
        <<< num_blocks, BLOCK_SIZE >>>(
                hornet,
                graph_offsets.size(),
                graph_offsets.data().get(),
                batch_offsets.data().get(),
                vertex_ids.data().get(),
                batch_edges,
                duplicate_flag.data().get());
}

template <typename CSoAPtrT, typename degree_t>
__global__
void write_unique_edges_kernel(
        CSoAPtrT in,
        CSoAPtrT out,
        const degree_t * __restrict__ offsets,
        const size_t num_elements) {
    size_t     id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (auto i = id; i < num_elements; i += stride) {
        if (offsets[i] != offsets[i+1]) {
            out[offsets[i]] = in[i];
        }
    }
}

template <typename CSoADataT, typename degree_t>
void write_unique_edges(
        CSoADataT& in,
        CSoADataT& out,
        thrust::device_vector<degree_t>& offsets) {
    auto in_ptr = in.get_soa_ptr();
    auto out_ptr = out.get_soa_ptr();
    const unsigned BLOCK_SIZE = 128;
    const size_t num_elements = offsets.size() - 1;
    write_unique_edges_kernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_elements), BLOCK_SIZE >>>
        (in_ptr, out_ptr, offsets.data().get(), num_elements);
}

template <typename HornetDeviceT, typename vid_t, typename degree_t, typename SoAPtrT>
__global__
void buildReallocateVerticesQueue(
        HornetDeviceT hornet,
        const vid_t * __restrict__ unique_sources,
        const degree_t * __restrict__ unique_degrees,
        const degree_t unique_sources_count,
        vid_t * __restrict__ realloc_sources,
        SoAPtrT realloc_vertex_access,
        SoAPtrT new_vertex_access,
        degree_t * __restrict__ realloc_sources_count,
        const bool is_insert,
        degree_t * __restrict__ graph_degrees) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    xlib::DeviceQueueOffset queue(realloc_sources_count);

    for (auto i = id; i < unique_sources_count; i += stride) {
        vid_t        src          = unique_sources[i];
        degree_t requested_degree = unique_degrees[i];
        auto vertex = hornet.vertex(src);

        degree_t old_degree = vertex.degree();

        if (graph_degrees != nullptr) { graph_degrees[i] = old_degree; }

        degree_t new_degree = is_insert ?
            old_degree + requested_degree :
            old_degree - requested_degree;

        bool realloc_flag = is_insert ?
            new_degree > vertex.limit() :
            new_degree <= (vertex.limit() / 2);

        if (realloc_flag) {
            int offset = queue.offset();
            realloc_sources[offset] = src;
            auto realloc_vertex_access_ref = realloc_vertex_access[offset];
            realloc_vertex_access_ref.template get<0>() = old_degree;
            realloc_vertex_access_ref.template get<1>() = vertex.edge_block_ptr();
            realloc_vertex_access_ref.template get<2>() = vertex.vertex_offset();
            realloc_vertex_access_ref.template get<3>() = vertex.edges_per_block();
            auto new_vertex_access_ref = new_vertex_access[offset];
            new_vertex_access_ref.template get<0>() = new_degree;
        } else {
            vertex.set_degree(new_degree);
        }
    }
}

template <int BLOCK_SIZE, typename HornetDeviceT, typename degree_t, typename SoAPtrT>
__global__
void move_adjacency_lists_kernel(
        HornetDeviceT hornet,
        SoAPtrT d_realloc_v_data,
        SoAPtrT d_new_v_data,
        const degree_t* __restrict__ graph_offsets,
        int graph_offsets_count) {
    using EdgePtrT = typename HornetDeviceT::VertexT::EdgeT::EdgeContainerT;

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&] (int pos, degree_t offset) {
        auto realloc_ref = d_realloc_v_data[pos];
        auto new_ref = d_new_v_data[pos];
        EdgePtrT r_eptr(realloc_ref. template get<1>(), realloc_ref. template get<3>());
        EdgePtrT n_eptr(new_ref. template get<1>(), new_ref. template get<3>());
        n_eptr[new_ref. template get<2>() + offset] = r_eptr[realloc_ref. template get<2>() + offset];
    };
    xlib::binarySearchLB<BLOCK_SIZE>(graph_offsets, graph_offsets_count, smem, lambda);
}

template <typename vid_t, typename degree_t, typename VAccessPtr, typename VMetaData>
__global__
void set_vertex_meta_data(
        vid_t * const realloc_src,
        VAccessPtr vertex_access_ptr,
        VMetaData d_new_v_data,
        const degree_t reallocated_vertices_count) {
    size_t     id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (auto i = id; i < reallocated_vertices_count; i += stride) {
        auto new_ref = vertex_access_ptr[realloc_src[i]];
        auto old_ref = d_new_v_data[i];
        new_ref.template get<0>() = old_ref.template get<0>();
        new_ref.template get<1>() = old_ref.template get<1>();
        new_ref.template get<2>() = old_ref.template get<2>();
        new_ref.template get<3>() = old_ref.template get<3>();
    }
}

template <int BLOCK_SIZE, typename HornetDeviceT, typename vid_t, typename degree_t, typename SoAPtrT>
__global__
void appendBatchEdgesKernel(
        HornetDeviceT hornet,
        const vid_t    * __restrict__ unique_src_ids,//
        const degree_t * __restrict__ batch_offsets,//
        const degree_t * __restrict__ old_degree,//
        const size_t batch_offsets_count,//
        SoAPtrT                          batch_edges) {

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&] (int pos, degree_t offset) {
        auto vertex = hornet.vertex(unique_src_ids[pos]);
        vertex.edge(old_degree[pos] + offset) = batch_edges[batch_offsets[pos] + offset];
    };
    xlib::binarySearchLB<BLOCK_SIZE>(batch_offsets, batch_offsets_count, smem, lambda);
}

template <int BLOCK_SIZE, typename HornetDeviceT, typename vid_t, typename degree_t>
__global__
void locate_erased_edges_kernel(
        HornetDeviceT hornet,//
        const size_t graph_offsets_count,//
        const degree_t * __restrict__ graph_offsets,//
        const degree_t * __restrict__ batch_src_offsets,//
        degree_t * __restrict__ batch_dst_degrees,//
        const vid_t    * __restrict__ unique_src_ids,//
        const vid_t    * __restrict__ batch_dst_ids,//
        degree_t * __restrict__ batch_erase_flag,
        degree_t * __restrict__ erase_edge_location) {

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&] (int pos, degree_t offset) {
                    auto     vertex = hornet.vertex(unique_src_ids[pos]);
                    assert(offset < vertex.degree());
                    auto e = vertex.edge(offset);
                    auto        dst = e.dst_id();
                    int start = batch_src_offsets[pos];
                    int end   = batch_src_offsets[pos + 1];
                    int found = xlib::lower_bound_left(
                            batch_dst_ids + start,
                            end - start,
                            dst);
                    if ((found >= 0) && (dst == batch_dst_ids[start + found])) {
                        //Write offset of graph edge
                        degree_t edge_count_offset = atomicSub(batch_dst_degrees + start + found, 1);
                        if (edge_count_offset > 0) {
                            erase_edge_location[edge_count_offset - 1 + start + found] = offset;
                            batch_erase_flag[edge_count_offset - 1 + start + found] = 1;
                        }
                    }
                };
    xlib::binarySearchLB<BLOCK_SIZE>(graph_offsets, graph_offsets_count, smem, lambda);
}

//Sets false to all locations in duplicatnEcorresponding batch_dst_ids
//is present in the graph
template <typename HornetDeviceT, typename vid_t, typename degree_t>
void locate_erased_edges(
        HornetDeviceT& hornet,
        thrust::device_vector<vid_t>& unique_sources,
        const vid_t * batch_dst_ids,
        thrust::device_vector<degree_t>& batch_src_offsets,
        thrust::device_vector<degree_t>& batch_dst_degrees,
        thrust::device_vector<degree_t>& graph_offsets,
        thrust::device_vector<degree_t>& batch_erase_flag,
        thrust::device_vector<degree_t>& erase_edge_location,
        const degree_t total_work) {
    const unsigned BLOCK_SIZE = 128;
    int smem = xlib::DeviceProperty::smem_per_block<degree_t>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(total_work, smem);
    batch_erase_flag.resize(batch_dst_degrees.size());
    erase_edge_location.resize(batch_dst_degrees.size());
    thrust::fill(batch_erase_flag.begin(), batch_erase_flag.end(), 0);
    thrust::fill(erase_edge_location.begin(), erase_edge_location.end(), 0);
    locate_erased_edges_kernel<BLOCK_SIZE>
        <<< num_blocks, BLOCK_SIZE >>>(
                hornet,
                graph_offsets.size(),
                graph_offsets.data().get(),
                batch_src_offsets.data().get(),
                batch_dst_degrees.data().get(),
                unique_sources.data().get(),
                batch_dst_ids,
                batch_erase_flag.data().get(),
                erase_edge_location.data().get());
}

template <typename vid_t, typename degree_t>
__global__
void markUniqueOffsetsKernel(
        vid_t * batch_src,
        vid_t * batch_dst,
        const degree_t nE,
        degree_t * edge_count,
        degree_t * offsets) {
    degree_t     id = blockIdx.x * blockDim.x + threadIdx.x;
    degree_t stride = gridDim.x * blockDim.x;

    for (degree_t i = id; i < nE; i += stride) {
        edge_count[i] = 0;
        if (i == 0) { offsets[0] = 0; continue; }
        bool discontinuity =
            (batch_src[i] != batch_src[i - 1]) ||
            (batch_dst[i] != batch_dst[i - 1]);
        offsets[i] = discontinuity? i : static_cast<degree_t>(0);
    }
}

template <typename degree_t>
__global__
void writeEdgeCountsKernel(
        const degree_t nE,
        degree_t * edge_count,
        degree_t * offsets) {
    degree_t     id = blockIdx.x * blockDim.x + threadIdx.x;
    degree_t stride = gridDim.x * blockDim.x;

    for (degree_t i = id; i < nE; i += stride) {
        degree_t self_offset = offsets[i];
        degree_t next_offset = nE;
        if (i < nE - 1) { next_offset = offsets[i + 1]; }
        if (self_offset != next_offset)
            edge_count[self_offset] = next_offset - self_offset;
    }
}

template <typename vid_t, typename degree_t>
void markUniqueOffsets(
        vid_t * batch_src,
        vid_t * batch_dst,
        const degree_t nE,
        thrust::device_vector<degree_t>& offsets,
        thrust::device_vector<degree_t>& edge_count,
        xlib::CubInclusiveMax<degree_t>& cub_prefixmax) {
    offsets.resize(nE);
    edge_count.resize(nE);
    const unsigned BLOCK_SIZE = 128;
    markUniqueOffsetsKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(nE), BLOCK_SIZE >>>
        (batch_src, batch_dst, nE,
         edge_count.data().get(), offsets.data().get());
    cub_prefixmax.run(offsets.data().get(), nE);
    writeEdgeCountsKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(nE), BLOCK_SIZE >>>
        (nE, edge_count.data().get(), offsets.data().get());
}

template <typename HornetDeviceT, typename vid_t, typename degree_t>
__global__
void overwriteDeletedEdgesKernel(
        HornetDeviceT hornet,
        const vid_t * __restrict__ sources,
        const degree_t * __restrict__ dst_offsets,
        const degree_t * __restrict__ src_offsets,
        const size_t sources_count) {
    size_t     id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (auto i = id; i < sources_count; i += stride) {
        auto vertex = hornet.vertex(sources[i]);
        vertex.edge(dst_offsets[i]) = vertex.edge(src_offsets[i]);
    }
}

template <typename T>
void print_arr(thrust::device_ptr<T> d, int count, std::string name) {
  std::cout<<"\n"<<name<<" : ";
  thrust::copy(d, d + count, std::ostream_iterator<T>(std::cout, " "));
}

template <typename T>
void print_arr(thrust::device_vector<T>& d, std::string name) {
  std::cout<<"\n"<<name<<" : ";
  thrust::copy(d.begin(), d.end(), std::ostream_iterator<T>(std::cout, " "));
}

template <typename HornetDeviceT, typename vid_t, typename degree_t>
void overwriteDeletedEdges(
        HornetDeviceT& hornet,
        thrust::device_vector<vid_t>& sources,
        thrust::device_vector<degree_t>& dst_offsets,
        thrust::device_vector<degree_t>& src_offsets) {
    const unsigned BLOCK_SIZE = 128;
    overwriteDeletedEdgesKernel<<<xlib::ceil_div<BLOCK_SIZE>(sources.size()), BLOCK_SIZE>>>(
            hornet,
            sources.data().get(),
            dst_offsets.data().get(),
            src_offsets.data().get(),
            sources.size());
    CHECK_CUDA_ERROR
}
