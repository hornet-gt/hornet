#include "GlobalSpace.cuh"                  //d_nV, d_nE
#include "Support/Device/SafeCudaAPI.cuh"   //cuMemcpyFromSymbol

namespace custinger_alg {
namespace detail {

template<void (*Operator)(int, void*)>
__global__ void forAllKernel(int num_items, void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < num_items; i += stride)
        Operator(i, optional_data);
}

} // namespace detail

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data) {
    forAllKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (num_items, optional_data);
}

//------------------------------------------------------------------------------
namespace detail {

template<void (*Operator)(custinger::vid_t, void*)>
__global__ void forAllnumVKernel(void* optional_data) {
    using custinger::vid_t;
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    vid_t size = static_cast<vid_t>(d_nV);

    for (vid_t i = id; i < size; i += stride)
        Operator(i, optional_data);
}

} // namespace detail

template<void (*Operator)(custinger::vid_t, void*)>
void forAllnumV(const custinger::cuStinger& custinger, void* optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    detail::forAllnumVKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (optional_data);
}

//------------------------------------------------------------------------------
namespace detail {

template<void (*Operator)(custinger::eoff_t, void*)>
__global__ void forAllnumEKernel(void* optional_data) {
    using custinger::eoff_t;
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    int  stride = gridDim.x * blockDim.x;
    eoff_t size = static_cast<eoff_t>(d_nE);

    for (eoff_t i = id; i < size; i += stride)
        Operator(i, optional_data);
}

} // namespace detail

template<void (*Operator)(custinger::eoff_t, void*)>
void forAllnumE(const custinger::cuStinger& custinger, void* optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    detail::forAllnumEKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (optional_data);
}

//==============================================================================
namespace detail {

template<void (*Operator)(Vertex, void*)>
__global__ void forAllVerticesKernel(void* optional_data) {
    using custinger::vid_t;
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    vid_t size = static_cast<vid_t>(d_nV);

    for (vid_t i = id; i < size; i += stride)
        Operator(Vertex(i), optional_data);
}

} // namespace detail

template<void (*Operator)(Vertex, void*)>
void forAllVertices(const custinger::cuStinger& custinger, void* optional_data){
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    detail::forAllVerticesKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (optional_data);
}

//------------------------------------------------------------------------------
namespace detail {

template<void (*Operator)(Vertex, void*)>
__global__ void forAllOutEdgesKernel(void* optional_data) {

}

inline void partition(const custinger::eoff_t* d_offsets,
                      int*& d_partitions, int&num_partitions) {
    /*using custinger::eoff_t;
    const unsigned SMEM_SIZE = xlib::SMemPerBlock<BLOCK_SIZE_OP1,eoff_t>::value;
    const unsigned PARTITION_SIZE = BLOCK_SIZE_OP1 * SMEM_SIZE;

    size_t nV = 0;
    cuMemcpyFromSymbol(d_nV, nV);
    num_partitions = xlib::ceil_div<PARTITION_SIZE>(nV);
    cuMalloc(d_partitions, num_partitions + 1);
    cuMalloc(d_offsets, nV);
    cuMemcpyToDevice(d_offsets, nV, d_offsets);
    static CuFreeAtExit<1>(d_offsets, d_partitions);

    xlib::blockPartition(d_offsets, nV, d_partitions, num_partitions);*/
}

} // namespace detail

template<void (*Operator)(Vertex, Edge, void*)>
void forAllEdges(const custinger::cuStinger& custinger,
                 const custinger::eoff_t* d_offsets, void* optional_data) {
    static int*  d_partitions = nullptr;
    static int num_partitions = 0;
    if (num_partitions == 0)
        detail::partition(d_offsets, d_partitions, num_partitions);

    detail::forAllOutEdgesKernel<Operator>
        <<< num_partitions, BLOCK_SIZE_OP1 >>> (d_partitions, optional_data);
}

//==============================================================================
/*namespace detail {

template<void (*Operator)(Vertex, void*)>
__global__ void forAllBatchEdgesKernel(EdgeBatch edge_batch,
                                       void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int   size = edge_batch.size;

    for (int i = id; i < size; i += stride)
        Operator(Vertex(i), Edge(i), optional_data);
}

} // namespace detail

template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllBatchEdges(const EdgeBatch& edge_batch, void* optional_data) {

}*/

} // namespace custinger_alg
