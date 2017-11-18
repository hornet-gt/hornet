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

int main(int argc, char* argv[]) {
    Timer<DEVICE> TM;

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
        (d_in, size, d_out);
}
