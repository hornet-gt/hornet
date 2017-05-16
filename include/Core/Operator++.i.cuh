#include "Core/cuStingerTypes.cuh"

namespace custinger_alg {
namespace detail {

template<typename Operator>
__global__ void forAllKernel(int size, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < size; i += stride)
        op(i);
}

template<typename Operator>
__global__ void forAllnumVKernel(custinger::vid_t d_nV, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (custinger::vid_t i = id; i < d_nV; i += stride)
        op(i);
}

template<typename Operator>
__global__ void forAllnumEKernel(custinger::eoff_t d_nE, Operator op) {
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    int  stride = gridDim.x * blockDim.x;

    for (custinger::eoff_t i = id; i < d_nE; i += stride)
        op(i);
}

template<typename Operator>
__global__ void forAllVerticesKernel(custinger::cuStingerDevData data,
                                     Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (custinger::vid_t i = id; i < data.nV; i += stride)
        op(custinger::Vertex(data, i));
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK, typename Operator>
__global__
void forAllEdgesKernel(const custinger::eoff_t* __restrict__ csr_offsets,
                       custinger::cuStingerDevData           data,
                       Operator                              op) {

    __shared__ custinger::degree_t smem[ITEMS_PER_BLOCK];
    const auto lambda = [&](int pos, custinger::degree_t offset) {
                                custinger::Vertex vertex(data, pos);
                                op(vertex, vertex.edge(offset));
                            };
    xlib::binarySearchLB<BLOCK_SIZE>(csr_offsets, data.nV + 1, smem, lambda);
}

} //namespace detail

//==============================================================================
//==============================================================================

template<typename Operator>
void forAll(size_t size, const Operator& op) {
    detail::forAllKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (size, op);
}

//------------------------------------------------------------------------------

template<typename Operator>
void forAllnumV(const custinger::cuStinger& custinger, const Operator& op) {
    detail::forAllnumVKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(custinger.nV()), BLOCK_SIZE_OP2 >>>
        (custinger.nV(), op);
}

//------------------------------------------------------------------------------

template<typename Operator>
void forAllnumE(const custinger::cuStinger& custinger, const Operator& op) {
    detail::forAllnumEKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(custinger.nE()), BLOCK_SIZE_OP2 >>>
        (custinger.nE(), op);
}

//------------------------------------------------------------------------------

template<typename Operator>
void forAllVertices(const custinger::cuStinger& custinger, const Operator& op) {
    detail::forAllVerticesKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(custinger.nV()), BLOCK_SIZE_OP2 >>>
        (custinger.device_data(), op);
}

//------------------------------------------------------------------------------

template<typename Operator>
void forAllEdges(custinger::cuStinger& custinger, const Operator& op) {
    using custinger::vid_t;
    const int PARTITION_SIZE = xlib::SMemPerBlock<BLOCK_SIZE_OP2, vid_t>::value;
    int num_partitions = xlib::ceil_div<PARTITION_SIZE>(custinger.nE());

    detail::forAllEdgesKernel<BLOCK_SIZE_OP2, PARTITION_SIZE, Operator>
       <<< num_partitions, BLOCK_SIZE_OP2 >>>
       (custinger.device_csr_offsets(), custinger.device_data(), op);
}

} // namespace custinger_alg
