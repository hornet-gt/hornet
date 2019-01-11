#include <iostream>
#include "Device/Primitives/WarpReduce.cuh"
#include "Device/Primitives/WarpScan.cuh"
//#include "Device/CubWrapper.cuh"
#include "Device/Util/SafeCudaAPI.cuh"
#include "Device/Util/PrintExt.cuh"
#include "StandardAPI.hpp"

using namespace hornets_nest;

struct __align__(16) SS {
     int a[2];
};

//__device__ DTuple<int, float> device_struct;

__device__ SS ss_struct;

__device__ std::tuple<int, float> tuple_struct;

__global__ void ptxKernel() {
    //DTuple<int, float> d_tuple = { 3, 2.0f };
    //device_struct = d_tuple;
    //device_struct = {1, 3.0};//make_devtuple(1,3.0);
    //device_struct = make_devtuple(3.0, 1);
    //ss_struct = { 1, 3.0 };
    //int AA[2] = { 2, 2 };
    //ss_struct = reinterpret_cast<SS&>(AA);
    //tuple_struct = { 1, 3.0f };
}

__device__ char* device_data;
__device__ int* device_data_int;

__global__ void ptxKernel1(int size) {
    int id = threadIdx.x;
    int stride = gridDim.x;

    for (int i = id; i < size; i += stride)
        device_data_int[i] = 3;
}

struct __align__(16) S {
    size_t aa;
    double k;
};

__device__ S* s_data;

__global__ void ptxKernel2(int size) {
    S s = {2, 3.0 };
    *s_data = s;
}

__device__ int val;

__global__ void ptxKernel3() {
    int a = 2;
    val = threadIdx.x * a;
}



struct A {
    __host__ __device__
    A() { printf("construct\n"); }

    __host__ __device__
    A(const A& obj) { printf("copy\n"); }

    __host__ __device__
    ~A() {
        printf("destroy\n");
    }
};

__global__ void classTest(A a) {
    printf("kernel\n");
}

__global__ void segReduceTest() {
    double value;
    //if (threadIdx.x < 8)
        value = 1;
    /*else if (threadIdx.x < 16)
        value = 2;
    else if (threadIdx.x < 24)
        value = 4;
    else if (threadIdx.x < 32)
        value = 8;*/
    //unsigned mask = 0b01010101010101010101010101010101;
    //xlib::WarpSegmentedReduce::add(value, mask);
    xlib::WarpExclusiveScan<4>::add(value);
    printf("%f ", value);
}

int exec(void) {
    segReduceTest<<<1, 32>>>();
    cudaDeviceSynchronize();
    std::cout << std::endl;

    return 0;
#if 0//Seunghwa Kang: this code does not execute and is not relevant to ptxtest, I may delete this sometime in the future.
    int* d_input, *d_output;
    int batch_size = 128;
    auto h_batch = new int[batch_size];
    std::fill(h_batch, h_batch + batch_size, 1);
    gpu::allocate(d_input, batch_size);
    gpu::allocate(d_output, batch_size);
    gpu::memsetZero(d_output, batch_size);
    host::copyToDevice(h_batch, batch_size, d_input);
    gpu::free(d_output, d_input);
#endif
}
int main() {
    int ret = 0;
#if defined(RMM_WRAPPER)
    gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec();

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

