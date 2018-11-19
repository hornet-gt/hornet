#include <XLib.hpp>
using namespace xlib;

__device__ int value[112] = {0, 0};
__device__ volatile int result;

__device__ char2 input[32];
__device__ int output[32];


__global__ void PTXoperation() {
    //int value2[8];// = {1,2};
    char4 value2[2];
    //auto value2 = reinterpret_cast<char (&)[5*4]>(value1);
    //char value2[8];
    /*value2[0] = threadIdx.x;
    value2[1] = threadIdx.x;
    value2[2] = threadIdx.x;
    value2[3] = threadIdx.x;
    value2[4] = threadIdx.x;
    value2[5] = threadIdx.x;
    value2[6] = threadIdx.x;
    value2[7] = threadIdx.x;*/


    value2[0] = make_char4(threadIdx.x, threadIdx.x, threadIdx.x, threadIdx.x);
    value2[1] = make_char4(threadIdx.x, threadIdx.x, threadIdx.x, threadIdx.x);


    //result = __bi(value[2], 3u);
    //reinterpret_cast<unsigned&>(word)
    #pragma unroll
    for (int i = 0; i < sizeof(value2); i++) {
        value[threadIdx.x * i] = value2[i].x;
        value[(i+1) * threadIdx.x] = value2[i].y;
        value[(i+2) * threadIdx.x] = value2[i].z;
        value[(i+3) * threadIdx.x] = value2[i].w;
    }
}

__global__ void loadOP() {
    /*output[0] = Load<DF>(input);
    output[1] = Load<CA>(input + 1);
    output[2] = Load<CG>(input + 2);
    output[3] = Load<CS>(input + 3);
    output[4] = Load<CV>(input + 4);
    output[5] = Load<NC>(input + 5);
    output[6] = Load<NC_CA>(input + 6);
    output[7] = Load<NC_CG>(input + 7);
    output[8] = Load<NC_CS>(input + 8);*/
}

__global__ void storeOP() {
    char2 aa = make_char2(1,1);
    Store<DF>(input, aa);
    Store<WB>(input + 1, aa);
    Store<CG>(input + 2, aa);
    Store<CS>(input + 3, aa);
    Store<CV>(input + 4, aa);
    //cub::ThreadStore<cub::STORE_CG>(input + 5, aa);
}

using _BITMASK_POLICY = BITMASK_POLICY<DF,        // Cache modifier for bit load
                                       DF,        // Cache modifier for bit store
                                       unsigned>;    // Size of access in bit     (8, 16, 32, 128)


__global__ void bitmask() {
    cuWriteBit<_BITMASK_POLICY>(value, 33);
    //cuWriteBit<_BITMASK_POLICY>(value, 32);
    //cuWriteBit<_BITMASK_POLICY>(value, 45);
}

__global__ void test_shuffle() {
    int value = threadIdx.x;
    int total = __shfl(value, 3);
    output[threadIdx.x] = value;
}

int main() {
    //PTXoperation<4><<<1, 1>>>();
    bitmask<<<1, 32>>>();

    int res[2];
    cudaMemcpyFromSymbol(res, value, sizeof(int) * 2);

    xlib::printBits(res, 64);
}
