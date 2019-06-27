#include "../SoA/SoAData.cuh"

namespace hornet {
namespace gpu {

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNETSTATIC::
HornetStatic(HORNETSTATIC::HInitT& h_init) noexcept :
    _nV(h_init.nV()),
    _nE(h_init.nE()),
    _id(_instance_count++),
    _vertex_data(h_init.nV()),
    _edge_data(xlib::upper_approx<512>(h_init.nE())) {
    initialize(h_init);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNETSTATIC::
initialize(HORNETSTATIC::HInitT& h_init) noexcept {
    SoAData<VertexTypes, DeviceType::HOST> vertex_data(h_init.nV());
    auto e_d = vertex_data.get_soa_ptr();

    xlib::byte_t * edge_block_ptr =
        reinterpret_cast<xlib::byte_t *>(_edge_data.get_soa_ptr().template get<0>());

    const auto * offsets = h_init.csr_offsets();
    for (int i = 0; i < h_init.nV(); ++i) {
        auto degree = offsets[i + 1] - offsets[i];
        auto e_ref = e_d[i];
        e_ref.template get<0>() = degree;
        e_ref.template get<1>() = edge_block_ptr;
        e_ref.template get<2>() = offsets[i];
        e_ref.template get<3>() = _edge_data.get_num_items();
    }
    _vertex_data.template copy(vertex_data);
    _edge_data.copy(h_init.edge_data_ptr(), DeviceType::HOST, h_init.nE());
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
HORNETSTATIC::
nV(void) const noexcept {
    return _nV;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
HORNETSTATIC::
nE(void) const noexcept {
    return _nE;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNETSTATIC::HornetDeviceT
HORNETSTATIC::
device(void) noexcept {
    return HornetDeviceT(_nV, _nE, _vertex_data.get_soa_ptr());
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNETSTATIC::
print(void) {
    SoAData<
        TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>,
        DeviceType::HOST> host_vertex_data(_vertex_data.get_num_items());
    host_vertex_data.copy(_vertex_data);
    auto ptr = host_vertex_data.get_soa_ptr();
    for (int i = 0; i < _nV; ++i) {
        degree_t v_degree = ptr[i].template get<0>();
        std::cout<<i<<" : "<<v_degree<<" | ";
        thrust::device_vector<degree_t> dst(v_degree);
        vid_t * dst_ptr = reinterpret_cast<vid_t*>(ptr[i].template get<1>()) + ptr[i].template get<2>();
        thrust::copy(dst_ptr, dst_ptr + v_degree, dst.begin());
        thrust::copy(dst.begin(), dst.end(), std::ostream_iterator<vid_t>(std::cout, " "));
        std::cout<<"\n";

    }
}

}
}
