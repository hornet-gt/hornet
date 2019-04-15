#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace hornet {
namespace gpu {
  template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
  vid_t Hornet<vid_t, TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>,degree_t>::max_degree_id() {
      auto start_ptr = _vertex_data.get_soa_ptr().template get<0>();
      int* iter = thrust::max_element(thrust::device, start_ptr, start_ptr + _nV);
      return (vid_t) (iter - start_ptr);
  }



}
}
