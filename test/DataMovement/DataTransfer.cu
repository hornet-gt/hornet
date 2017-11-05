#include <iostream>
#include <climits>
#include "../../XLib.hpp"
using namespace timer_cuda;

using namespace data_movement;

//------------------------------------------------------------------------------
// ORDERED : REG_TO_GLOBAL
//------------------------------------------------------------------------------

template<int STEPS, typename T>
__global__ void WarpOrdered_RegToGlobal(T* vector) {
    __shared__ T SMem[ SMem_Per_Block<T, 32>::value ];
    T TMP[STEPS];
    #pragma unroll
    for (int i = 0; i < STEPS; i++)
        TMP[i] = STEPS * threadIdx.x + i;

    T* SMemThread = SMem + LaneID() * STEPS;
    T* SMemT = SMem;
    warp::computeGlobalOffset<STEPS>(SMemT, vector);

    warp_ordered_adv::RegToGlobal(TMP, SMemT, SMemThread, vector);
}

//==============================================================================
//------------------------------------------------------------------------------
// ORDERED : REG_TO_GLOBAL
//------------------------------------------------------------------------------

template<int STEPS, typename T>
__global__ void WarpOrdered_GlobalToReg(T* vector) {
    __shared__ T SMem[ SMem_Per_Block<T, 32>::value ];
    T TMP[STEPS];

    T* SMemThread = SMem + LaneID() * STEPS;
    T* SMemT = SMem;
    warp::computeGlobalOffset<STEPS>(SMemT, vector);

    warp_ordered_adv::GlobalToReg(vector, SMemT, SMemThread, TMP);

    for (int K = 0; K < 32; K++) {
        if (K == threadIdx.x) {
            for (int i = 0; i < STEPS; i++)
                printf("%d ", TMP[i]);
            printf("\n");
        }
    }
}


/*
__global__ void warpOrderedSharedToReg(int* vector) {
    __shared__ int Value[32 * 6];
    int Queue[6];

    for (int i = threadIdx.x; i < 32 * 6; i += 32)
        Value[i] = i;

    warp::ordered::SharedToReg(Value, Queue);

    if (threadIdx.x == 1) {
        for (int i = 0; i < 6; i++) {
            printf("%d ", Queue[i]);
        }
        printf("\n");
    }
}

__global__ void warpUnorderedSharedToReg(int* vector) {
    __shared__ int Value[32 * 6];
    int Queue[6];

    for (int i = threadIdx.x; i < 32 * 6; i += 32)
        Value[i] = i;

    warp::unordered::SharedToReg(Value, Queue);

    if (threadIdx.x == 1) {
        for (int i = 0; i < 6; i++) {
            printf("%d ", Queue[i]);
        }
        printf("\n");
    }
}



//------------------------------------------------------------------------------

__global__ void warpOrderedRegToShared(int* vector) {
    __shared__ int Value[32 * 6];
    int Queue[6];

    for (int i = 0; i < 6; i++)
        Queue[i] = i;

    warp::ordered::RegToShared(Queue, Value);

    if (threadIdx.x == 0) {
        for (int i = 0; i < 32 * 6; i++) {
            printf("%d ", Value[i]);
        }
        printf("\n");
    }
}

__global__ void warpUnorderedRegToShared(int* vector) {
    __shared__ int Value[32 * 6];
    int Queue[6];

    for (int i = 0; i < 6; i++)
        Queue[i] = i;

    warp::unordered::RegToShared(Queue, Value);

    if (threadIdx.x == 0) {
        for (int i = 0; i < 32 * 6; i++) {
            printf("%d ", Value[i]);
        }
        printf("\n");
    }
}*/

//------------------------------------------------------------------------------



int main() {
    const int STEPS = 4;
    const int SIZE = 32 * STEPS;
    int* vector;
    cudaMalloc(&vector, SIZE * sizeof(int4));
    cudaMemset(vector, 0, SIZE * sizeof(int4));
    const int GRIDDIM = 1;
    using T = int;

    WarpOrdered_RegToGlobal<STEPS, T>
                    <<<GRIDDIM, 32>>>(reinterpret_cast<T*>(vector));

    CUDA_ERROR("WarpOrdered_RegToGlobal")

    T host_vector[SIZE];
    cudaMemcpy(host_vector, vector, SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < SIZE; i++) {
        if (host_vector[i] != i)
            ERROR("WarpOrdered_RegToGlobal");
    }
    std::cout << std::endl << "-> WarpOrdered_RegToGlobal passed" << std::endl << std::endl;


    WarpOrdered_GlobalToReg<STEPS, T>
                    <<<GRIDDIM, 32>>>(reinterpret_cast<T*>(vector));
    CUDA_ERROR("WarpOrdered_GlobalToReg")
}
