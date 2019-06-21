#ifndef STATIC_CUH
#define STATIC_CUH

#include "../Conf/Common.cuh"
#include "../Conf/HornetConf.cuh"
#include "../HornetDevice/HornetDevice.cuh"
#include "../HornetInitialize/HornetInit.cuh"
#include "../BatchUpdate/BatchUpdate.cuh"
#include "../MemoryManager/BlockArray/BlockArray.cuh"
#include "../Hornet.cuh"
#include <map>

namespace hornet {

template <DeviceType = DeviceType::DEVICE, typename = VID_T, typename = EMPTY, typename = DEGREE_T> class COO;
template <DeviceType = DeviceType::DEVICE, typename = VID_T, typename = EMPTY, typename = DEGREE_T> class CSR;
template <DeviceType = DeviceType::DEVICE, typename = EMPTY, typename = DEGREE_T> class VertexMeta;

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
class COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t> {
  template <typename, typename, typename, typename> friend class Hornet;
  template <DeviceType, typename, typename, typename> friend class COO;
  template <typename, typename, typename> friend class BatchUpdate;

  SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, device_t> _edge;

public:
  template <typename T>
  using Vector = typename
  std::conditional<
  (device_t == DeviceType::DEVICE),
  typename thrust::device_vector<T>,
  typename thrust::host_vector<T>>::type;

  template <DeviceType other_device>
  COO(COO<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& other);

  template <DeviceType other_device>
  COO(COO<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>&& other);

  COO(SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::HOST>&& other);
  COO(const degree_t edgeCount = 0);

  void resize(const degree_t size) noexcept;

  SoAPtr<vid_t, vid_t, EdgeMetaTypes...> getPtr(void) noexcept;
  vid_t* srcPtr(void) noexcept;
  vid_t* dstPtr(void) noexcept;

  template<unsigned N>
  typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes*...>::type>::type
  edgeMetaPtr() noexcept;

  template<unsigned N>
  typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes const*...>::type>::type
  edgeMetaPtr() const noexcept;

  void append(const COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& other) noexcept;

  template <DeviceType other_device>
  void copy(const COO<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& other) noexcept;

  void gather(COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& other,
      const Vector<degree_t>& map) noexcept;

  void sort(void) noexcept;

  degree_t size(void) noexcept;
};

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
std::multimap<vid_t, TypeList<vid_t, EdgeMetaTypes...>> getHostMMap(
    COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& coo) {
  auto *src = coo.srcPtr();
  auto ptr = coo.getPtr();
  COO<DeviceType::HOST, vid_t, TypeList<EdgeMetaTypes...>, degree_t> host_coo;
  if (device_t == DeviceType::DEVICE) {
    host_coo.resize(coo.size());
    host_coo.copy(coo);
    src = host_coo.srcPtr();
    ptr = host_coo.getPtr();
  }
  auto eptr = ptr.get_tail();
  std::multimap<vid_t, TypeList<vid_t, EdgeMetaTypes...>> mmap;
  for (int i = 0; i < coo.size(); ++i) {
    mmap.insert(std::make_pair(src[i], hornet::getTuple(eptr[i])));
  }
  return mmap;
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
class CSR<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t> {
  template <DeviceType, typename, typename, typename> friend class CSR;

  SoAData<TypeList<vid_t, EdgeMetaTypes...>, device_t> _index;

public:

  template <typename T>
  using Offset = typename
  std::conditional<
  (device_t == DeviceType::DEVICE),
  typename thrust::device_vector<T>,
  typename thrust::host_vector<T>>::type;

private:

  Offset<degree_t> _offset;

public:

  template <DeviceType other_device>
  CSR(CSR<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>&& other);
  CSR(Offset<vid_t>&& offset, SoAData<TypeList<vid_t, EdgeMetaTypes...>, DeviceType::HOST>&& other_index);

  CSR(const degree_t edgeCount = 0, const degree_t vertexCount = 0);

  void resize(const degree_t vertices, const degree_t edges) noexcept;

  degree_t* offset(void) noexcept;
  vid_t*     index(void) noexcept;

  template<unsigned N>
  typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes*...>::type>::type
  edgeMetaPtr() noexcept;

  template<unsigned N>
  typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes const*...>::type>::type
  edgeMetaPtr() const noexcept;
};

template <typename... VertexMetaTypes, typename degree_t, DeviceType device_t>
class VertexMeta<device_t, TypeList<VertexMetaTypes...>, degree_t> {
  template <DeviceType, typename, typename> friend class VertexMeta;

  SoAData<TypeList<VertexMetaTypes...>, device_t> _data;

  public:
  template <DeviceType other_device>
  VertexMeta(SoAData<TypeList<VertexMetaTypes...>, other_device>& other);

  template <DeviceType other_device>
  VertexMeta(SoAData<TypeList<VertexMetaTypes...>, other_device>&& other);

  template <DeviceType other_device>
  VertexMeta(VertexMeta<other_device, TypeList<VertexMetaTypes...>, degree_t>&& other);

  VertexMeta(const degree_t length = 0);

  template<unsigned N>
  typename std::enable_if<(N < (sizeof...(VertexMetaTypes))), typename xlib::SelectType<N, VertexMetaTypes*...>::type>::type
  vertexMetaPtr() noexcept;

  template<unsigned N>
  typename std::enable_if<(N < (sizeof...(VertexMetaTypes))), typename xlib::SelectType<N, VertexMetaTypes const*...>::type>::type
  vertexMetaPtr() const noexcept;
};

#define HCOO COO<device_t, vid_t, \
                      TypeList<EdgeMetaTypes...>,\
                      degree_t>

#define HCSR CSR<device_t, vid_t, \
                      TypeList<EdgeMetaTypes...>,\
                      degree_t>

#define VMETA VertexMeta<device_t, \
                      TypeList<VertexMetaTypes...>, \
                      degree_t>

}

#include "Static.i.cuh"

#endif
