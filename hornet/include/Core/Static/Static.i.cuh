namespace hornet {

////////////////////////////////COO////////////////////////////////
template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template <DeviceType other_device>
HCOO::COO(COO<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>&& other) : _edge(std::move(other._edge)) {}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template <DeviceType other_device>
HCOO::COO(COO<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& other) : _edge(other.size()) {
_edge.copy(other._edge); }

//template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
//HCOO::COO(COO<DeviceType::HOST,   vid_t, TypeList<EdgeMetaTypes...>, degree_t>&& other) : _edge(std::move(other._edge)) {}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
HCOO::COO(SoAData<TypeList<vid_t, vid_t, EdgeMetaTypes...>, DeviceType::HOST>&& other) : _edge(std::move(other)) {}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
HCOO::COO(const degree_t edgeCount) : _edge(edgeCount) {}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
void HCOO::
resize(const degree_t size) noexcept {
  _edge.resize(size);
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
SoAPtr<vid_t, vid_t, EdgeMetaTypes...> HCOO::
getPtr(void) noexcept {
  return _edge.get_soa_ptr();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
vid_t* HCOO::
srcPtr(void) noexcept {
  return _edge.get_soa_ptr().template get<0>();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
vid_t* HCOO::
dstPtr(void) noexcept {
  return _edge.get_soa_ptr().template get<1>();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template<unsigned N>
typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes*...>::type>::type
HCOO::edgeMetaPtr() noexcept {
  return _edge.get_soa_ptr().template get<N+2>();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template<unsigned N>
typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes const*...>::type>::type
HCOO::edgeMetaPtr() const noexcept {
  return _edge.get_soa_ptr().template get<N+2>();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
void HCOO::
append(const HCOO& other) noexcept {
  _edge.append(other._edge);
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template <DeviceType other_device>
void HCOO::
copy(const COO<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& other) noexcept {
  _edge.copy(other._edge);
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
void HCOO::
gather(HCOO& other, const Vector<degree_t>& map) noexcept {
  //_edge.gather(other._edge, map);
  RecursiveGather<0, (2 + sizeof...(EdgeMetaTypes))>::assign(other.getPtr(), getPtr(), map, static_cast<degree_t>(map.size()));
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
void HCOO::
sort(void) noexcept {
  _edge.sort();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
degree_t HCOO::
size(void) noexcept {
  return _edge.get_num_items();
}

////////////////////////////////COO////////////////////////////////

////////////////////////////////CSR////////////////////////////////
template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template <DeviceType other_device>
HCSR::CSR(CSR<other_device, vid_t, TypeList<EdgeMetaTypes...>, degree_t>&& other) :
  _offset(std::move(other._offset)), _index(std::move(other._index))  {}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
HCSR::CSR(HCSR::Offset<vid_t>&& offset,
    SoAData<TypeList<vid_t, EdgeMetaTypes...>, DeviceType::HOST>&& other_index) :
  _offset(std::move(_offset)), _index(std::move(_index))  {}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
HCSR::CSR(const degree_t edgeCount, const degree_t vertexCount) :
  _index(edgeCount), _offset(vertexCount) {}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
void HCSR::
resize(const degree_t edgeCount, const degree_t vertexCount) noexcept {
  _index.resize(edgeCount);
  _offset.resize(vertexCount);
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
degree_t* HCSR::
offset(void) noexcept {
  return _offset.data().get();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
vid_t* HCSR::
index(void) noexcept {
  return _index.get_soa_ptr().template get<0>();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template<unsigned N>
typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes*...>::type>::type
HCSR::edgeMetaPtr() noexcept {
  return _index.get_soa_ptr().template get<N+1>();
}

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
template<unsigned N>
typename std::enable_if<(N < (sizeof...(EdgeMetaTypes))), typename xlib::SelectType<N, EdgeMetaTypes const*...>::type>::type
HCSR::edgeMetaPtr() const noexcept {
  return _index.get_soa_ptr().template get<N+1>();
}
////////////////////////////////CSR////////////////////////////////

///////////////////////////////VMeta///////////////////////////////
template <typename... VertexMetaTypes, typename degree_t, DeviceType device_t>
template <DeviceType other_device>
VMETA::VertexMeta(SoAData<TypeList<VertexMetaTypes...>, other_device>& other) :
  _data(other) {}

template <typename... VertexMetaTypes, typename degree_t, DeviceType device_t>
template <DeviceType other_device>
VMETA::VertexMeta(SoAData<TypeList<VertexMetaTypes...>, other_device>&& other) :
  _data(std::move(other)) {}

template <typename... VertexMetaTypes, typename degree_t, DeviceType device_t>
template <DeviceType other_device>
VMETA::VertexMeta(VertexMeta<other_device, TypeList<VertexMetaTypes...>, degree_t>&& other) :
  _data(std::move(other._data)) {}

template <typename... VertexMetaTypes, typename degree_t, DeviceType device_t>
VMETA::VertexMeta(const degree_t length) : _data(length) {}

template <typename... VertexMetaTypes, typename degree_t, DeviceType device_t>
template<unsigned N>
typename std::enable_if<(N < (sizeof...(VertexMetaTypes))), typename xlib::SelectType<N, VertexMetaTypes*...>::type>::type
VMETA::
vertexMetaPtr() noexcept {
  return _data.get_soa_ptr().template get<N>();
}

template <typename... VertexMetaTypes, typename degree_t, DeviceType device_t>
template<unsigned N>
typename std::enable_if<(N < (sizeof...(VertexMetaTypes))), typename xlib::SelectType<N, VertexMetaTypes const*...>::type>::type
VMETA::
vertexMetaPtr() const noexcept {
  return _data.get_soa_ptr().template get<N>();
}

///////////////////////////////VMeta///////////////////////////////

}
