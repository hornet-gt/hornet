
namespace cu_stinger {
/*
template<int INDEX, typename T>
struct Recursive {
    HOST_DEVICE static void op() {
        T<INDEX>();
        Recursive<INDEX>::op();
    }
};
template<typename T>
struct Recursive<-1, T> {
    HOST_DEVICE static void op() {}
};*/
/*
template<int INDEX>
struct AA {
    AA() {
        _ptrs[i] = d_ptrs[i + 3] + index * EXTRA_VTYPE_SIZE[i];
    }
};*/

__device__ __forceinline__
Vertex::Vertex(id_t index) noexcept {
    xlib::SeqDev<VextexSizes> VTYPE_SIZE_D;
    _ptrs[0] = d_ptrs[0] + index * VTYPE_SIZE_D[0];   //degree
    _ptrs[1] = d_ptrs[1] + index * VTYPE_SIZE_D[1];   //limit
    auto ptr = d_ptrs[2] + index * VTYPE_SIZE_D[2];   //edge
    _ptrs[2] = reinterpret_cast<byte_t*>(*reinterpret_cast<edge_t**>(ptr));
    //#pragma unroll
    //for (int i = 0; i < NUM_EXTRA_VTYPES; i++)      //Fused multiply–add
    //    _ptrs[i + 3] = d_ptrs[i + 3] + index * VTYPE_SIZE2[i];//EXTRA_VTYPE_SIZE[i];
}

__device__ __forceinline__
degree_t Vertex::degree() const noexcept {
    return *reinterpret_cast<degree_t*>(_ptrs[0]);
}

__device__ __forceinline__
Edge Vertex::edge(degree_t index) const noexcept {
    return Edge(_ptrs[2], index);
}

//==============================================================================

__device__ __forceinline__
Edge::Edge(byte_t* block_ptr, off_t index) noexcept {
     _ptrs[0] = block_ptr + index * sizeof(id_t);
    //#pragma unroll
    //for (int i = 0; i < NUM_ETYPES; i++)
    //    _ptrs[i] = block_ptr + index * ETYPE_SIZE[i];       //Fused multiply–add
}

__device__ __forceinline__
id_t Edge::dst() const noexcept {
    return *reinterpret_cast<id_t*>(_ptrs[0]);
}

template<typename T>
__device__ __forceinline__
typename std::tuple_element<(NUM_ETYPES > 1 ? 1 : 0), edge_t>::type
Edge::weight() const noexcept {
    static_assert(!std::is_same<T, void>::value,
                  "weight is not part of edge type list");
    const int N = NUM_ETYPES > 1 ? 1 : 0;
    using     R = typename std::tuple_element<N, edge_t>::type;
    return *reinterpret_cast<R*>(_ptrs[N]);
}

//==============================================================================

__device__ __forceinline__
Vertex VertexSet::operator[](id_t index) const noexcept {
    return Vertex(index);
}

} // namespace cu_stinger
