#include "XLib.hpp"
using namespace timer;

struct DEV_CONST {
    int* __restrict__ ptr;
    int* __restrict__ ptr1;
    int* __restrict__ ptr2;
    int* __restrict__ ptr3;
};

__constant__ DEV_CONST dev_const;

__constant__ int* const_ptr;

__device__ __forceinline__ void write() {
    for (int i = 0; i < 100; i++)
        dev_const.ptr[i] = dev_const.ptr3[i] ;
}


__global__ void use_ptr() {
    write();
}

__global__ void use_ptr2(int* __restrict__ dev_const) {
    for (int i = 0; i < 100; i++)
        dev_const[i] = i;
}

__global__ void use_ptr3() {
    for (int i = 0; i < 100; i++)
        const_ptr[i] = i;
}

__device__ int Array[256];

__global__ void PTXoperation() {
    Array[0] = (int) threadIdx.x % 16;
    Array[1] = (int) threadIdx.x % 16u;
    Array[2] = (int) threadIdx.x & 15;

    Array[3] = (int) threadIdx.x / 16;
    Array[4] = (int) threadIdx.x / 16u;
    Array[5] = (int) threadIdx.x >> 4;
}

__device__ int2 devInt2[10];

__global__ void regOperation() {
    int2 f = devInt2[1];
    devInt2[0] = f;
}


__global__ void regOperation2() {
    int2 f = devInt2[1];
    devInt2[0].x = f.x;
    devInt2[0].y = f.y;
}

const int SIZE = 1 << 22;
__device__ int ArrayIN[SIZE];
__device__ int ArrayOUT[SIZE];

template<unsigned SPLIT = 1>
__global__ void storeKernel() {
    int* ptrOUT = ArrayOUT;
    int end = SIZE;

    if (SPLIT != 1) {
        ptrOUT += (threadIdx.x / (32 / SPLIT)) * (SIZE / SPLIT);
        end = ((threadIdx.x / (32 / SPLIT)) + 1) * (SIZE / SPLIT);
    }

    for (int i = threadIdx.x % (32 / SPLIT); i < end; i += 32 / SPLIT)
        ArrayOUT[i] = 1;
}

int main() {
    int* ptr_host;
    cudaMalloc(&ptr_host, 100 * sizeof(int));

    DEV_CONST dev_const_h;
    dev_const_h.ptr = ptr_host;
    cudaMemcpyToSymbol(dev_const, &dev_const_h, sizeof(DEV_CONST));


    CUDA_ERROR("dd");

    PTXoperation<<<1, 1>>>();
    Timer_cuda TM;
    TM.start();

    storeKernel<<<1, 32>>>();

    TM.getTime();
    TM.start();

    storeKernel<2><<<1, 32>>>();

    TM.getTime("split = 16");
}
