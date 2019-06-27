namespace hornet {
namespace gpu {

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNET::HornetDeviceT
HORNET::
device(void) noexcept {
    return HornetDeviceT(_nV, _nE, _vertex_data.get_soa_ptr());
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNET::
insert(BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>& batch, bool removeBatchDuplicates, bool removeGraphDuplicates) {
    auto hornet_device = device();
    //Preprocess batch according to user preference
    batch.preprocess(
            hornet_device, removeBatchDuplicates, removeGraphDuplicates);

    _nE = _nE + batch.nE();

    reallocate_vertices(batch, true);

    batch.appendBatchEdges(hornet_device);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNET::
reallocate_vertices(gpu::BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>& batch,
        const bool is_insert) {
    if (batch.nE() == 0) { return; }
    SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t> h_realloc_v_data;
    SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t> h_new_v_data;
    SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t> d_realloc_v_data;
    SoAPtr<degree_t, xlib::byte_t*, degree_t, degree_t> d_new_v_data;
    degree_t reallocated_vertices_count = 0;

    //Get list of vertices that need to be reallocated
    //realloc_vertex_meta_data contains old adjacency list information. This is used by appendBatchEdges.
    //new_vertex_meta_data contains buffer to store new adjacency list information from block array manager calls below

    auto hornet_device = device();
    batch.get_reallocate_vertices_meta_data(
            hornet_device, h_realloc_v_data, h_new_v_data, d_realloc_v_data, d_new_v_data, reallocated_vertices_count, is_insert);

    CUDA_CHECK_LAST()
    for (degree_t i = 0; i < reallocated_vertices_count; i++) {
        auto ref = h_new_v_data[i];
        auto access_data = _ba_manager.insert(ref.template get<0>());
        ref.template get<1>() = access_data.edge_block_ptr;;
        ref.template get<2>() = access_data.vertex_offset;
        ref.template get<3>() = access_data.edges_per_block;
    }

    ////Move adjacency list and edit vertex access data
    batch.move_adjacency_lists(hornet_device, _vertex_data.get_soa_ptr(), h_realloc_v_data, h_new_v_data, d_realloc_v_data, d_new_v_data, reallocated_vertices_count, is_insert);

    CUDA_CHECK_LAST()
    for (degree_t i = 0; i < reallocated_vertices_count; i++) {
        auto ref = h_realloc_v_data[i];
        _ba_manager.remove(ref.template get<0>(), ref.template get<1>(), ref.template get<2>());
    }
    CUDA_CHECK_LAST()

}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNET::
erase(gpu::BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>& batch, bool removeBatchDuplicates) {
    auto hornet_device = device();
    //Preprocess batch according to user preference
    //std::cout<<"\nBEFORE DELETE\n";
    //print();
    batch.preprocess_erase(hornet_device, removeBatchDuplicates);
    CHECK_CUDA_ERROR
    _nE = _nE - batch.nE();
    //std::cout<<"\nBEFORE REALLOCATE\n";
    //print();
    reallocate_vertices(batch, false);
    CHECK_CUDA_ERROR
    //std::cout<<"\nAFTER REALLOCATE\n";
    //print();
}

}
}
