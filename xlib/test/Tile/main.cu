#include <iostream>
#include <random>
#include "Device/DataMovement/Tile.cuh"
#include "Device/Util/Timer.cuh"
#include "Device/Util/SafeCudaAPI.cuh"
#include "Device/Util/PTX.cuh"
#include "Device/Util/SafeCudaAPISync.cuh"
#include "cublas_v2.h"

#define SAFE_CUBLAS_CALL(a) if(a != CUBLAS_STATUS_SUCCESS) {                    \
                                printf ("CUBLAS initialization failed\n");      \
                                return EXIT_FAILURE;                                \
                            }

using namespace timer;

const unsigned BLOCK_SIZE = 256;
const unsigned     UNROLL = 1;
using VType = int4;

template<typename T>
__global__
void copyKernel(const T* __restrict__ d_in, int size, T* d_out) {
    using LoadTileT  = LoadTile <BLOCK_SIZE, T, VType, UNROLL>;
    using StoreTileT = StoreTile<BLOCK_SIZE, T, VType, UNROLL>;
    //using  LoadTileT = IlLoadTile <BLOCK_SIZE, T, int4, UNROLL>;
    //using StoreTileT = StoreTile<BLOCK_SIZE, T, int4, UNROLL * 2>;

    LoadTileT  load_tile(d_in, size);
    StoreTileT store_tile(d_out, size);

    while (load_tile.is_valid()) {
        T array[LoadTileT::THREAD_ITEMS];
        load_tile.load(array);
        store_tile.store(array);
    }

    int id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for (int i = load_tile.last_index() + id; i < size; i += load_tile.stride())
        d_out[i] = d_in[i];
}

template<typename T>
__global__
void workStructKernel(const T* __restrict__ d_in, int size, T* d_out) {
    xlib::WordArray<int, 10> test(d_out);
    d_out[0] = test[size];
}

template<typename T>
__global__
void copyTest1(const T* __restrict__ d_in, int size, T* __restrict__ d_out) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride)
        d_out[i] = d_in[i];
}

template<typename T, typename R>
__global__
void copyTest2(const T* __restrict__ d_in1,
               const R* __restrict__ d_in2,
               int size,
               T* __restrict__ d_out1,
               R* __restrict__ d_out2) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride) {
        d_out1[i] = d_in1[i];
        d_out2[i] = d_in2[i];
    }
}

template<typename T, typename R>
__global__
void copyTest3(const T* __restrict__ d_in1,
               const T* __restrict__ d_in2,
               const R* __restrict__ d_in3,
               int size,
               T* __restrict__ d_out1,
               T* __restrict__ d_out2,
               R* __restrict__ d_out3) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride) {
        d_out1[i] = d_in1[i];
        d_out2[i] = d_in2[i];
        d_out3[i] = d_in3[i];
    }
}

int main(int argc, char* argv[]) {
    /*sTimer<DEVICE> TM;

    using T = int;
    const int size = (1 << 29);

    auto h_array = new T[size];
    std::iota(h_array, h_array + size, 0);

    T* d_in, *d_out;
    cuMalloc(d_in, size);
    cuMalloc(d_out, size);
    cuMemcpyToDevice(h_array, size, d_in);
    cuMemset0x00(d_out, size);
    //--------------------------------------------------------------------------
    TM.start();

    const int MUL = (sizeof(VType) / sizeof(int)) * UNROLL;
    copyKernel
        <<< xlib::ceil_div<BLOCK_SIZE * MUL>(size), BLOCK_SIZE >>>
    //copyKernel <<< xlib::ResidentBlocks<BLOCK_SIZE>::value, BLOCK_SIZE >>>
        (d_in, size, d_out);

    CHECK_CUDA_ERROR
    TM.stop();
    TM.print("copy");

    cuMemcpyToHost(d_out, size, h_array);
    for (int i = 0; i < size; i++) {
        if (h_array[i] != i)
            ERROR("Wrong result at: ", i, "  value: ", h_array[i]);
            //std::cout << "Wrong result at: " << i << "  value: " << h_array[i] << std::endl;
    }
    std::cout << "Correct <>" << std::endl;

    cublasHandle_t handle;
    SAFE_CUBLAS_CALL( cublasCreate(&handle) )
    TM.start();

    SAFE_CUBLAS_CALL( cublasScopy(handle, size,
                                  reinterpret_cast<float*>(d_in), 1,
                                  reinterpret_cast<float*>(d_out), 1) )

    TM.stop();
    TM.print("cublas");
    SAFE_CUBLAS_CALL( cublasDestroy(handle) )

    workStructKernel
        <<< xlib::ceil_div<BLOCK_SIZE * MUL>(size), BLOCK_SIZE >>>
    //copyKernel <<< xlib::ResidentBlocks<BLOCK_SIZE>::value, BLOCK_SIZE >>>
        (d_in, size, d_out);*/

#define TEST1
    //__________________________________________________________________________
    int size = (1 << 30);

#if defined(TEST3)
    int16_t *d_in1, *d_in2, *d_out1, *d_out2;
    int8_t *d_in3, *d_out3;
    cuMalloc(d_in1, size);
    cuMalloc(d_in2, size);
    cuMalloc(d_in3, size);
    cuMalloc(d_out1, size);
    cuMalloc(d_out2, size);
    cuMalloc(d_out3, size);
#elif defined(TEST1)
    int* d_in4, *d_out4;
    cuMalloc(d_in4, size);
    cuMalloc(d_out4, size);
#elif defined (TEST2)
    int16_t* d_in5, *d_out5;
    int8_t *d_in6, *d_out6;
    cuMalloc(d_in5, size);
    cuMalloc(d_out5, size);
    cuMalloc(d_in6, size);
    cuMalloc(d_out6, size);
#endif

    Timer<DEVICE> TM;

    for (int i = (1 << 16); i < size; i *= 2) {
        TM.start();

#if defined(TEST3)
        copyTest3 <<< xlib::ceil_div<BLOCK_SIZE>(i), BLOCK_SIZE >>>
            (d_in1, d_in2, d_in3, i, d_out1, d_out2, d_out3);
#elif defined (TEST2)
        copyTest2 <<< xlib::ceil_div<BLOCK_SIZE>(i), BLOCK_SIZE >>>
            (d_in5, d_in6, i, d_out5, d_out6);
#elif defined(TEST1)
        copyTest1 <<< xlib::ceil_div<BLOCK_SIZE>(i), BLOCK_SIZE >>>
            (d_in4, i, d_out4);
#endif

        TM.stop();
        std::cout << "i:\t" << i << "\t" << TM.duration() << std::endl;
    }
}
