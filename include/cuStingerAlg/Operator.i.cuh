#include "Core/cuStingerTypes.cuh"          //cu_stinger::Vertex
#include "Csr/CsrTypes.cuh"                 //csr::Vertex
#include "GlobalSpace.cuh"                  //d_nV, d_nE
#include "Support/Device/SafeCudaAPI.cuh"   //cuMemcpyFromSymbol

using cu_stinger::Vertex;

namespace cu_stinger_alg {

const int BLOCK_SIZE_OP = 256;

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
        <<< xlib::ceil_div<BLOCK_SIZE_OP>(num_items), BLOCK_SIZE_OP >>>
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


//==============================================================================
//==============================================================================
//==============================================================================
/////////////////
/// C++11 API ///
/////////////////

template<typename Lambda>
__global__ void forAllKernel(int size, Lambda lambda) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < size; i += stride)
        lambda(i);
}

template<typename Lambda>
void forAll(size_t size, Lambda lambda) {
    forAllKernel<<< xlib::ceil_div<BLOCK_SIZE_OP>(size), BLOCK_SIZE_OP >>>
        (size, lambda);
}

template<typename Lambda>
void forAllnumV(Lambda lambda) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAll(num_items, lambda);
}

template<typename Lambda>
void forAllnumE(Lambda lambda) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAll(num_items, lambda);
}

//------------------------------------------------------------------------------

template<typename Lambda>
void forAllVertices(Lambda lambda) {

}

template<typename Lambda>
void forAllEdges(Lambda lambda) {

}




} // namespace cu_stinger_alg
