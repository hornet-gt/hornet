#include <iostream>
#include <chrono>
#include <functional>
#include <random>
#include <limits>

#include "XLib.hpp"
using namespace xlib;

__global__ void threadDefault(int* DataIN, int* DataOUT) {
    int Local_data[32];
    for (int i = 0; i < 32; i++)
        Local_data[i] = ThreadLoad<>::OP(DataIN + i);

    for (int i = 0; i < 32; i++)
        ThreadStore<>::OP(DataOUT + i, Local_data[i]);
}

__global__ void threadCA(volatile int* DataIN, int* DataOUT) {
    int Local_data[32];
    for (int i = 0; i < 32; i++)
        Local_data[i] = ThreadLoad<LOAD_CA>::OP(DataIN + i);

    for (int i = 0; i < 32; i++)
        ThreadStore<STORE_WB>::OP(DataOUT + i, Local_data[i]);
}

__global__ void threadChar(volatile unsigned char* DataIN, volatile unsigned char* DataOUT) {
    unsigned char Local_data[32];
    for (int i = 0; i < 32; i++)
        Local_data[i] = ThreadLoad<LOAD_CA>::OP(DataIN + i);

    for (int i = 0; i < 32; i++)
        ThreadStore<STORE_WB>::OP(DataOUT + i, Local_data[i]);
}

__global__ void threadChar2(char2* DataIN, char2* DataOUT) {
    char2 Local_data[16];
    for (int i = 0; i < 16; i++)
        Local_data[i] = ThreadLoad<LOAD_CA>::OP(DataIN + i);

    for (int i = 0; i < 16; i++)
        ThreadStore<STORE_WB>::OP(DataOUT + i, Local_data[i]);
}

__global__ void threadChar4(uchar4* DataIN, uchar4* DataOUT) {
    uchar4 Local_data[8];
    for (int i = 0; i < 8; i++)
        Local_data[i] = ThreadLoad<LOAD_CA>::OP(DataIN + i);

    for (int i = 0; i < 8; i++)
        ThreadStore<STORE_WB>::OP(DataOUT + i, Local_data[i]);
}



int main() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(-1432435, 23423533);

    const int INPUT_SIZE = 32;
    int DataIN[INPUT_SIZE];
    int DataOUT[INPUT_SIZE];
    int* devDataIN, *devDataOUT;
    __SAFE_CALL( cudaMalloc(&devDataIN, sizeof(DataIN)) );
    __SAFE_CALL( cudaMalloc(&devDataOUT, sizeof(DataIN)) );

    for (int i = 0; i < INPUT_SIZE; i++)
        DataIN[i] = distribution(generator);

    xlib::printArray(DataIN);

    __SAFE_CALL( cudaMemcpy(devDataIN, DataIN, sizeof(DataIN),
                 cudaMemcpyHostToDevice) );

    threadDefault<<<1, 1>>>(devDataIN, devDataOUT);

    __SAFE_CALL( cudaMemcpy(DataOUT, devDataOUT, sizeof(DataOUT),
                 cudaMemcpyDeviceToHost) );

    if (!std::equal(DataIN, DataIN + INPUT_SIZE, DataOUT))
        ERROR("Default copy")

    //--------------------------------------------------------------------------
    unsigned char DataINchar[INPUT_SIZE];
    unsigned char DataOUTchar[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++)
            DataINchar[i] = static_cast<unsigned char>(distribution(generator));

    xlib::printArray(DataINchar);

    unsigned char* devDataINchar, *devDataOUTchar;
    __SAFE_CALL( cudaMalloc(&devDataINchar, sizeof(DataINchar)) );
    __SAFE_CALL( cudaMalloc(&devDataOUTchar, sizeof(DataINchar)) );
    __SAFE_CALL( cudaMemcpy(devDataINchar, DataINchar, sizeof(DataINchar),
                 cudaMemcpyHostToDevice) );
    __SAFE_CALL( cudaMemset(devDataOUTchar, 0x0, sizeof(DataOUTchar)) );

    threadChar<<<1, 1>>>(devDataINchar, devDataOUTchar);

    __SAFE_CALL( cudaMemcpy(DataOUTchar, devDataOUTchar, sizeof(DataOUTchar),
                 cudaMemcpyDeviceToHost) );

    if (!std::equal(DataINchar, DataINchar + INPUT_SIZE, DataOUTchar))
        ERROR("Char copy")

        __SAFE_CALL( cudaMemset(devDataOUTchar, 0x0, sizeof(DataOUTchar)) );

    //--------------------------------------------------------------------------

    __SAFE_CALL( cudaMemset(devDataOUTchar, 0x0, sizeof(DataOUTchar)) );

    threadChar2<<<1, 1>>>(reinterpret_cast<char2*>(devDataINchar),
                         reinterpret_cast<char2*>(devDataOUTchar));

    __SAFE_CALL( cudaMemcpy(DataOUTchar, devDataOUTchar, sizeof(DataOUTchar),
                 cudaMemcpyDeviceToHost) );

    if (!std::equal(DataINchar, DataINchar + INPUT_SIZE, DataOUTchar))
        ERROR("Char2 copy")

    //--------------------------------------------------------------------------

    __SAFE_CALL( cudaMemset(devDataOUTchar, 0x0, sizeof(DataOUTchar)) );

    threadChar4<<<1, 1>>>(reinterpret_cast<uchar4*>(devDataINchar),
                         reinterpret_cast<uchar4*>(devDataOUTchar));

    __SAFE_CALL( cudaMemcpy(DataOUTchar, devDataOUTchar, sizeof(DataOUTchar),
                 cudaMemcpyDeviceToHost) );

    if (!std::equal(DataINchar, DataINchar + INPUT_SIZE, DataOUTchar))
        ERROR("Char2 copy")

    //--------------------------------------------------------------------------

    for (int i = 0; i < INPUT_SIZE; i++)
            DataIN[i] = distribution(generator);

    xlib::printArray(DataIN);

    __SAFE_CALL( cudaMemcpy(devDataIN, DataIN, sizeof(DataIN),
                 cudaMemcpyHostToDevice) );
    __SAFE_CALL( cudaMemset(devDataOUT, 0x0, sizeof(DataOUT)) );

    threadCA<<<1, 1>>>(devDataIN, devDataOUT);

    __SAFE_CALL( cudaMemcpy(DataOUT, devDataOUT, sizeof(DataOUT),
                 cudaMemcpyDeviceToHost) );

    if (!std::equal(DataIN, DataIN + INPUT_SIZE, DataOUT))
        ERROR("CA copy")
}


//sdfsdf
