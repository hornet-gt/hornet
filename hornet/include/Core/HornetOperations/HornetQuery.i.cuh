#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "../SoA/SoAData.cuh"

namespace hornet {
namespace gpu {
  template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
  vid_t
  HORNET::
  max_degree_id() const noexcept {
      auto start_ptr = _vertex_data.get_soa_ptr().template get<0>();
      auto* iter = thrust::max_element(thrust::device, start_ptr, start_ptr + _nV);
      if (iter == start_ptr + _nV) {
          return static_cast<vid_t>(-1);
      } else {
          return static_cast<vid_t>(iter - start_ptr);
      }
  }

  template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
  degree_t
  HORNET::
  max_degree() const noexcept {
      auto start_ptr = _vertex_data.get_soa_ptr().template get<0>();
      auto* iter = thrust::max_element(thrust::device, start_ptr, start_ptr + _nV);
      if (iter == start_ptr + _nV) {
          return static_cast<degree_t>(0);
      } else {
          degree_t d;
          cudaMemcpy(&d, iter, sizeof(degree_t), cudaMemcpyDeviceToHost);
          return d;
      }
  }

  //template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
  //std::pair<SoAData<degree_t, VertexMetaTypes...>,
  //  SoAData<vid_t*, EdgeMetaTypes...>>
  //HORNET::
  //getCSR(bool sortAdjacencyList) const noexcept {
  //    return std::make_tuple(0, 0);
  //}

  template <int BLOCK_SIZE, typename HornetDeviceT, typename degree_t, typename SoAPtrT>
  __global__
  void flattenCOOKernel(
        HornetDeviceT hornet,
        degree_t* offsets,
        SoAPtrT ptr
      ) {
    const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&] (int pos, degree_t eOffset) {
        auto vertex = hornet.vertex(pos);
        auto *src = ptr.template get<0>();
        auto e = ptr.get_tail();
        e[offsets[pos] + eOffset] = vertex.edge(eOffset);
        src[offsets[pos] + eOffset] = pos;
    };

    xlib::binarySearchLB<BLOCK_SIZE>(offsets, hornet.nV() + 1, smem, lambda);
  }

  template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
  SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE>
  HORNET::
  getCOO(bool sortAdjacencyList) noexcept {
    //allocate src, dst
    //allocate degree
    //scan vertex degrees
    //load balanced copy to src, dst
    HornetDeviceT hornet_device = device();
    SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE> coo;
    if (nE() == 0) { return coo; }
    else { coo.resize(nE()); }

    thrust::device_vector<degree_t> degree(nV() + 1);
    auto start_ptr = _vertex_data.get_soa_ptr().template get<0>();
    thrust::copy(thrust::device, start_ptr, start_ptr + _nV, degree.begin());
    thrust::exclusive_scan(thrust::device, degree.begin(), degree.end(), degree.begin());

    const int BLOCK_SIZE = 256;
    int smem = xlib::DeviceProperty::smem_per_block<degree_t>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(nE(), smem);
    flattenCOOKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(hornet_device, degree.data().get(), coo.get_soa_ptr());

    if (sortAdjacencyList) {
      SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::DEVICE> out_coo(nE());
      thrust::device_vector<degree_t> range;
      if (sort_batch(coo.get_soa_ptr(), nE(), range, out_coo.get_soa_ptr())) { return out_coo;
      } else { return coo; }
    } else {
      return coo;
    }
  }


}
}
