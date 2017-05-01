#include "GlobalSpace.cuh"                  //d_nV, d_nE
#include "Support/Device/SafeCudaAPI.cuh"   //cuMemcpyFromSymbol

namespace cu_stinger_alg {

template<void (*Operator)(int, void*)>
__global__ void forAllKernel(int num_items, void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < num_items; i += stride)
        Operator::apply(i, optional_data);
}

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data) {
    forAllKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP>(num_items), BLOCK_SIZE_OP >>>
        (num_items, optional_data...);
}

//------------------------------------------------------------------------------

template<void (*Operator)(vid_t, void*)>
__global__ void forAllnumVKernel(void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (vid_t i = id; i < d_nV; i += stride)
        Operator::apply(i, optional_data);
}

template<void (*Operator)(vid_t, void*)>
void forAllnumV(void* optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAllnumVKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP>(num_items), BLOCK_SIZE_OP >>>
        (optional_data);
}

//------------------------------------------------------------------------------

template<void (*Operator)(eoff_t, void*)>
__global__ void forAllnumEKernel(void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (eoff_t i = id; i < d_nE; i += stride)
        Operator::apply(i, optional_data);
}

template<void (*Operator)(eoff_t, void*)>
void forAllnumE(void* optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAllnumEKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP>(num_items), BLOCK_SIZE_OP >>>
        (optional_data);
}

//==============================================================================

template<typename Operator, typename... TArgs>
__global__ void forAllVerticesKernel(TArgs... args) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < d_nV; i += stride)
        Operator::apply(Vertex(i), args...);
}

template<typename Operator, typename... TArgs>
void forAllVertices(TArgs... optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAllVerticesKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP>(num_items), BLOCK_SIZE_OP >>>
        (optional_data...);
}

template<typename Operator, typename... TArgs>
void forAllEdges(TArgs... optional_data) {

}

//==============================================================================

template<typename Operator, typename... TArgs>
void forAllBatchEdges(TArgs... optional_data) {

}

} // namespace cu_stinger_alg
