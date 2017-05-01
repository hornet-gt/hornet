#include "GlobalSpace.cuh"                  //d_nV, d_nE
#include "Support/Device/SafeCudaAPI.cuh"   //cuMemcpyFromSymbol

namespace cu_stinger_alg {

template<void (*Operator)(int, void*)>
__global__ void forAllKernel(int num_items, void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < num_items; i += stride)
        Operator(i, optional_data);
}

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data) {
    forAllKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (num_items, optional_data...);
}

//------------------------------------------------------------------------------

template<void (*Operator)(vid_t, void*)>
__global__ void forAllnumVKernel(void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    vid_t size = static_cast<vid_t>(d_nV);

    for (vid_t i = id; i < size; i += stride)
        Operator(i, optional_data);
}

template<void (*Operator)(vid_t, void*)>
void forAllnumV(void* optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAllnumVKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (optional_data);
}

//------------------------------------------------------------------------------

template<void (*Operator)(eoff_t, void*)>
__global__ void forAllnumEKernel(void* optional_data) {
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    int  stride = gridDim.x * blockDim.x;
    eoff_t size = static_cast<eoff_t>(d_nE);

    for (eoff_t i = id; i < size; i += stride)
        Operator(i, optional_data);
}

template<void (*Operator)(eoff_t, void*)>
void forAllnumE(void* optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAllnumEKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (optional_data);
}

//==============================================================================

template<void (*Operator)(Vertex, void*)>
__global__ void forAllVerticesKernel(void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    vid_t size = static_cast<vid_t>(d_nV);

    for (vid_t i = id; i < size; i += stride)
        Operator(Vertex(i), optional_data);
}

template<void (*Operator)(Vertex, void*)>
void forAllVertices(void* optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAllVerticesKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (optional_data);
}

//------------------------------------------------------------------------------

template<int SIZE>
struct CuFreeAtExit {
    template<typename... TArgs>
    explicit CuFreeAtExit(TArgs... args) noexcept : _tmp {{ args... }} {}

    ~CuFreeAtExitAtExit() _tmp {
        for (const auto& it : tmp)
            cuFree(it);
    }
private:
    std::array<void*, SIZE> _tmp;
};

template<void (*Operator)(Vertex, Edge, void*)>
void forAllEdges(const eoff_t* csr_offsets, void* optional_data) {
    const unsigned SMEM_SIZE = xlib::SMemPerBlock<BLOCK_SIZE_OP1,eoff_t>::value;
    const unsigned PARTITION_SIZE = BLOCK_SIZE_OP1 * SMEM_SIZE;
    static size_t nV = 0;
    if (nV == 0) {
        cuMemcpyFromSymbol(d_nV, nV);
        int num_partitions = xlib::ceil_div<PARTITION_SIZE>(nV);
        int* d_partitions;
        cuMalloc(d_partitions, num_partitions);
        xlib::blockPartition(csr_offsets, nV, d_partitions, num_partitions);
        static CuFreeAtExit<1>(d_partitions);
        flag = false;
    }
    forAllOutEdgesKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(nV), BLOCK_SIZE_OP1 >>>
        (optional_data)
}

//==============================================================================

template<typename Operator, typename... TArgs>
void forAllBatchEdges(TArgs... optional_data) {

}

} // namespace cu_stinger_alg
