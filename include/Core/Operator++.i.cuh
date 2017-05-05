#include "Support/Device/SafeCudaAPI.cuh"   //cuMemcpyFromSymbol

namespace custinger_alg {
/////////////////
/// C++11 API ///
/////////////////

template<typename Operator>
__global__ void forAllKernel(int size, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < size; i += stride)
        op(i);
}

template<typename Operator>
void forAll(size_t size, Operator op) {
    forAllKernel <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (size, op);
}

//------------------------------------------------------------------------------

template<typename Operator>
__global__ void forAllnumVKernel(Operator op) {
    using custinger::vid_t;
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    vid_t size = static_cast<vid_t>(d_nV);

    for (vid_t i = id; i < size; i += stride)
        op(i);
}

template<typename Operator>
void forAllnumV(Operator op) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAllnumVKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(num_items), BLOCK_SIZE_OP2 >>> (op);
}

//------------------------------------------------------------------------------

template<typename Operator>
__global__ void forAllnumEKernel(Operator op) {
    using custinger::eoff_t;
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    int  stride = gridDim.x * blockDim.x;
    eoff_t size = static_cast<eoff_t>(d_nE);

    for (eoff_t i = id; i < size; i += stride)
        op(i);
}

template<typename Operator>
void forAllnumE(Operator op) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAllnumEKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(num_items), BLOCK_SIZE_OP2 >>> (op);
}

//------------------------------------------------------------------------------

template<typename Operator>
__global__ void forAllVerticesKernel(Operator op) {
    using custinger::vid_t;
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    vid_t size = static_cast<vid_t>(d_nV);

    for (vid_t i = id; i < size; i += stride)
        op(Vertex(i));
}

template<typename Operator>
void forAllVertices(Operator op) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAllVerticesKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(num_items), BLOCK_SIZE_OP2 >>> (op);
}

//------------------------------------------------------------------------------

template<typename Operator>
void forAllEdges(Operator op) {

}

} // namespace custinger_alg
