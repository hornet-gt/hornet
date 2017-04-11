/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include "Support/Algorithm.hpp"
#include <iomanip>
#include <string>

namespace xlib {

//to update
template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool cuEqual(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B) {
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    CUDA_ERROR("Copy To Host");

    bool flag = xlib::equal<FAULT>(start_A, end_A, ArrayCMP);
    delete[] ArrayCMP;
    return flag;
}

//to update
template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool cuEqual(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
             bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type)) {

    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    CUDA_ERROR("Copy To Host");

    bool flag = xlib::equal<FAULT>(start_A, end_A, ArrayCMP, equalFunction);
    delete[] ArrayCMP;
    return flag;
}

//to update
template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool cuEqualSorted(iteratorA_t start_A, iteratorA_t end_A,
                   iteratorB_t start_B) {
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    CUDA_ERROR("Copy To Host");

    bool flag = xlib::equalSorted<FAULT>(start_A, end_A, ArrayCMP);
    delete[] ArrayCMP;
    return flag;
}

template<class T>
inline unsigned gridConfig(T FUN, unsigned block_size,
                           unsigned dyn_shared_mem, int problem_size) {
    int num_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, FUN,
                                                  static_cast<int>(block_size),
                                                  dyn_shared_mem);
    return static_cast<unsigned>(
        std::min(deviceProperty::getNum_of_SMs() * num_blocks, problem_size));
}

//to update
template<typename T>
__global__ void scatter(const int* __restrict__ toScatter, int scatter_size,
                        T* __restrict__ Dest, T value) {

    unsigned ID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = ID; i < scatter_size; i += blockDim.x * gridDim.x)
        Dest[ toScatter[i] ] = value;
}

//to update
template<typename T>
__global__ void fill(T* devArray, int fill_size, T value) {
    unsigned ID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = ID; i < fill_size; i += blockDim.x * gridDim.x)
        devArray[ i ] = value;
}


//to update
template <typename T>
__global__ void fill(T* devMatrix, int n_of_rows, int n_of_columns,
                     const T value, int integer_pitch) {

    const int X = blockDim.x * blockIdx.x + threadIdx.x;
    const int Y = blockDim.y * blockIdx.y + threadIdx.y;
    if (integer_pitch == 0)
        integer_pitch = n_of_columns;

    for (int i = Y; i < n_of_rows; i += blockDim.y * gridDim.y) {
        for (int j = X; j < n_of_columns; j += blockDim.x * gridDim.x)
            devMatrix[i * integer_pitch + j] = value;
    }
}

/*
template <unsigned UNROLLING, typename T>
__global__ void copy_unroll(T* __restrict__ Input,
                     const int size,
                     T* __restrict__ Output) {

    const unsigned LOCAL_SIZE = WARP_SIZE * UNROLLING;
    unsigned warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    unsigned warp_offset = warp_id * LOCAL_SIZE + LaneID();
    unsigned stride = ((blockDim.x * gridDim.x) / WARP_SIZE) * UNROLLING;

    T* __restrict__ Output_ptr = Output + warp_offset;
    T* __restrict__ Input_ptr = Input + warp_offset;

    for (int i = warp_id; i < size / LOCAL_SIZE; i += stride) {
        #pragma unroll
        for (int j = 0; j < UNROLLING; j++)
            Output_ptr[j * WARP_SIZE] = Input_ptr[j * WARP_SIZE];
        Output_ptr += stride;
    }
    //--------------------------------------------------------------------------
    unsigned global_id = blockDim.x * blockIdx.x + threadIdx.x;
    stride = blockDim.x * gridDim.x;
    Output_ptr = Output + lowerApprox<LOCAL_SIZE>(size);
    Input_ptr = Input + lowerApprox<LOCAL_SIZE>(size);

    for (int i = global_id; i < size % LOCAL_SIZE; i += stride)
        Output_ptr[i] = Input_ptr[i];
}*/

//to update
template <typename T>
__global__ void copy(T* __restrict__ Input,
                     const int size,
                     T* __restrict__ Output) {

    unsigned global_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;

    for (int i = global_id; i < size; i += stride)
        Output[i] = Input[i];
        //Output[i] = __ldg(Input + i);
}

} // namespace xlib
