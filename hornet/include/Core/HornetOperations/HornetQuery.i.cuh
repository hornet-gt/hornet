#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace hornet {
namespace gpu {
  template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
  vid_t Hornet<vid_t, TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>,degree_t>::max_degree_id() const noexcept {
      auto start_ptr = _vertex_data.get_soa_ptr().template get<0>();
      auto* iter = thrust::max_element(thrust::device, start_ptr, start_ptr + _nV);
      if (iter == start_ptr + _nV) {
          return static_cast<vid_t>(-1);
      } else {
          return static_cast<vid_t>(iter - start_ptr);
      }
  }

  template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
  degree_t Hornet<vid_t, TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>,degree_t>::max_degree() const noexcept {
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



}
}
