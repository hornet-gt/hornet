#include "cuStingerAlgConfig.cuh"
#include "GlobalSpace.cuh"

namespace cu_stinger_alg {

template<typename Operator, typename T, typename... TArgs>
__global__ void forAllKernel(T* array, int num_items, TArgs... args) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < num_items; i += stride)
        Operator::apply(array[i], args...);
}

template<typename Operator, typename T, typename... TArgs>
void forAll(T* d_array, int num_items, TArgs... optional_data) {
    forAllKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE >>>
        (d_array, num_items, optional_data...);
}

//------------------------------------------------------------------------------

template<typename Operator, typename T, typename... TArgs>
void forAllnumV(T* d_array, TArgs... optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAll<Operator>(d_array, num_items, optional_data...);
}

//------------------------------------------------------------------------------

template<typename Operator, typename T, typename... TArgs>
void forAllnumE(T* d_array, TArgs... optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAll<Operator>(d_array, num_items, optional_data...);
}

//==============================================================================

template<typename Operator, typename... TArgs>
__global__ void forAllVerticesKernel(TArgs... args) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //for (int i = id; i < d_nV; i += stride)
    //    Operator::apply(Vertex(i), args...);
}

template<typename Operator, typename... TArgs>
void forAllVertices(TArgs... optional_data) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAllVerticesKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE >>>
        (optional_data...);
}

template<typename Operator, typename... TArgs>
void forAllEdges(TArgs... optional_data) {

}

//==============================================================================

template<typename Operator, typename... TArgs>
void forAllBatchEdges(TArgs... optional_data) {

}

template<typename Operator, typename T, typename... TArgs>
void forAllTraverseEdges(Queue<T> queue, TArgs... optional_data) {

}

template<typename Operator, typename T, typename... TArgs>
void forAllTraverseEdges(T* d_array, int num_items, TArgs... optional_data) {

}

} // namespace cu_stinger_alg
