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
#include "../SoA/SoAData.cuh"

namespace hornet {
namespace gpu {


template<unsigned N, unsigned SIZE>
struct AssignData {
    template <typename... EdgeMetaTypes, typename vid_t, typename degree_t>
    static void assign(
            CSoAPtr<vid_t, EdgeMetaTypes...> e_ptr, degree_t vertex_block_offset,
            SoAPtr<vid_t const, EdgeMetaTypes const...> edge_data,
            degree_t vertex_csr_offset, const degree_t vertex_degree) {
        if (edge_data.template get<N>() != nullptr) {
            for (degree_t i = 0; i < vertex_degree; ++i) {
                e_ptr[vertex_block_offset + i].template get<N>() =
                    edge_data[vertex_csr_offset + i].template get<N>();
            }
        }
        AssignData<N+1, SIZE>::assign(e_ptr, vertex_block_offset, edge_data, vertex_csr_offset, vertex_degree);
    }
};

template<unsigned N>
struct AssignData<N, N> {
    template <typename... EdgeMetaTypes, typename vid_t, typename degree_t>
    static void assign(
            CSoAPtr<vid_t, EdgeMetaTypes...> e_ptr, degree_t vertex_block_offset,
            SoAPtr<vid_t const, EdgeMetaTypes const...> edge_data,
            degree_t vertex_csr_offset, const degree_t vertex_degree) { }
};

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
int HORNET::_instance_count = 0;

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNET::
Hornet(void) noexcept { }

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNET::
Hornet(degree_t nV) noexcept :
    _nV(nV),
    _nE(0),
    _id(_instance_count++),
    _vertex_data(nV) { }

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNET::
Hornet(HORNET::HInitT& h_init) noexcept :
    _nV(h_init.nV()),
    _nE(h_init.nE()),
    _id(_instance_count++),
    _vertex_data(h_init.nV()) {
    initialize(h_init);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNET::
initialize(HornetInit<
    vid_t,
    TypeList<VertexMetaTypes...>,
    TypeList<EdgeMetaTypes...>, degree_t>& h_init) noexcept {

    SoAData<VertexTypes, DeviceType::HOST> vertex_data(h_init.nV());
    auto e_d = vertex_data.get_soa_ptr();

    std::unordered_map<xlib::byte_t*, HostBlockArray> h_blocks;

    const auto * offsets = h_init.csr_offsets();
    for (int i = 0; i < h_init.nV(); ++i) {
        auto degree = offsets[i + 1] - offsets[i];
        auto device_ad = _ba_manager.insert(degree);
        auto e_ref = e_d[i];
        e_ref.template get<0>() = degree;
        e_ref.template get<1>() = device_ad.edge_block_ptr;
        e_ref.template get<2>() = device_ad.vertex_offset;
        e_ref.template get<3>() = device_ad.edges_per_block;

        CSoAPtr<vid_t, EdgeMetaTypes...> e_ptr;

        auto search = h_blocks.find(device_ad.edge_block_ptr);
        if (search != h_blocks.end()) {
            e_ptr = CSoAPtr<vid_t, EdgeMetaTypes...>(search->second.get_blockarray_ptr(), device_ad.edges_per_block);
        } else {
            HostBlockArray new_block_array(
                    1<<xlib::ceil_log2(degree),
                    device_ad.edges_per_block);
            e_ptr = CSoAPtr<vid_t, EdgeMetaTypes...>(new_block_array.get_blockarray_ptr(), device_ad.edges_per_block);
            h_blocks.insert(std::make_pair(device_ad.edge_block_ptr, std::move(new_block_array)));
        }
        AssignData<0, (1 + sizeof...(EdgeMetaTypes))>::assign(
                e_ptr, device_ad.vertex_offset,
                h_init.edge_data_ptr(), offsets[i], degree);
    }

    _vertex_data.template copy(vertex_data);
    for(auto &b : h_blocks) {
        DeviceCopy::copy(
                b.second.get_blockarray_ptr(), DeviceType::HOST,
                b.first, DeviceType::DEVICE,
                b.second.mem_size());
    }

}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNET::
print(void) {
    SoAData<
        TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>,
        DeviceType::HOST> host_vertex_data(_vertex_data.get_num_items());
    host_vertex_data.copy(_vertex_data);
    auto ptr = host_vertex_data.get_soa_ptr();
    for (int i = 0; i < _nV; ++i) {
        degree_t v_degree = ptr[i].template get<0>();
        std::cout<<i<<" : "<<v_degree<<" | ";
        if (v_degree != 0) {
          thrust::device_vector<degree_t> dst(v_degree);
          vid_t * dst_ptr = reinterpret_cast<vid_t*>(ptr[i].template get<1>()) + ptr[i].template get<2>();
          thrust::copy(dst_ptr, dst_ptr + v_degree, dst.begin());
          thrust::copy(dst.begin(), dst.end(), std::ostream_iterator<vid_t>(std::cout, " "));
        }
        std::cout<<"\n";

    }
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
HORNET::
nV(void) const noexcept {
    return _nV;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
HORNET::
nE(void) const noexcept {
    return _nE;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNET::
reset(HornetInit<
    vid_t,
    TypeList<VertexMetaTypes...>,
    TypeList<EdgeMetaTypes...>, degree_t>& h_init) noexcept {
  _nV = h_init.nV();
  _nE = h_init.nE();
  SoAData<
      TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>,
      DeviceType::DEVICE> new_vertex_data(h_init.nV());
  _vertex_data = std::move(new_vertex_data);
  _ba_manager.removeAll();
  initialize(h_init);
}

}
}
