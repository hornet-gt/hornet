#include "cuStingerAlg/cuStingerAlgConfig.cuh"
#include "GlobalSpace.cuh"

#pragma once

namespace cu_stinger_alg {

template<typename Operator, typename T>
void forAll(T* d_array, int num_items);

template<typename Operator, typename T>
void forAllnumV(T* d_array);

template<typename Operator, typename T>
void forAllnumE(T* d_array);

//------------------------------------------------------------------------------

template<typename Operator, typename T>
void forAllVertices(T optional_data);

template<typename Operator, typename T>
void forAllEdges(T optional_data);

//------------------------------------------------------------------------------

template<typename Operator, typename T>
void forAllBatchEdges(T optional_data);

template<typename Operator, typename T>
void forAllVerticesTraverse(T optional_data);

//==============================================================================

//Static Algorithms Interface
class StaticAlgorithm {
public:
	virtual void init(const cu_stinger::cuStinger& custing) = 0;
	virtual void run()                          = 0;
    virtual void reset()                        = 0;
	virtual void release()                      = 0;
};

//------------------------------------------------------------------------------

template<typename Operator, typename T>
__global__ void forAllKernel(T* array, int num_items) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < num_items; i += stride)
        Operator::apply(array[i]);
}

template<typename Operator, typename T>
void forAll(T* d_array, int num_items) {
    forAllKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE >>>
        (d_array, num_items);
}

//------------------------------------------------------------------------------

template<typename Operator, typename T>
void forAllnumV(T* d_array) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAll<Operator>(d_array, num_items);
}

} // namespace cu_stinger_alg
