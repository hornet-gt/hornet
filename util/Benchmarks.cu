#include <Support/Device/Timer.cuh>
#include <Support/Device/SafeCudaAPI.cuh>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace timer;
using byte_t = char;

int main() {
    size_t size = 1024;
    Timer<DEVICE> TM;

    std::vector<float> malloc_time;
    std::vector<float> mallochost_time;
    std::vector<float> dev2dev_time;
    std::vector<float> memset_time;

    while (true) {
        byte_t* d_array;
        if (cudaMalloc(&d_array, size) != cudaSuccess)
            break;

        TM.start();

        auto h_array = new byte_t[size];
        cuMemcpyToDevice(h_array, size, d_array);
        delete[] h_array;

        TM.stop();
        malloc_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        TM.start();

        byte_t* h_array_pinned;
        cudaMallocHost(&h_array_pinned, size);
        cuMemcpyToDevice(h_array_pinned, size, d_array);
        cudaFreeHost(h_array_pinned);

        TM.stop();
        mallochost_time.push_back(TM.duration());

        cuFree(d_array);
        size *= 2;
    }
    //==========================================================================
    std::vector<float> malloc_HtD_time;
    std::vector<float> mallochost_HtD_time;

    size = 1024;
    while (true) {
        byte_t* d_array;
        if (cudaMalloc(&d_array, size) != cudaSuccess)
            break;

        auto h_array = new byte_t[size];
        TM.start();

        cuMemcpyToDevice(h_array, size, d_array);

        TM.stop();
        delete[] h_array;
        malloc_HtD_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        byte_t* h_array_pinned;
        cudaMallocHost(&h_array_pinned, size);
        TM.start();

        cuMemcpyToDevice(h_array_pinned, size, d_array);

        TM.stop();
        cudaFreeHost(h_array_pinned);
        mallochost_HtD_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        TM.start();

        cudaMemset(d_array, 0x00, size);

        TM.stop();
        memset_time.push_back(TM.duration());

        //----------------------------------------------------------------------
        byte_t* d_array2;
        cudaMalloc(&d_array2, size);
        TM.start();

        cudaMemcpy(d_array2, d_array, size,  cudaMemcpyDeviceToDevice);

        TM.stop();
        dev2dev_time.push_back(TM.duration());

        cuFree(d_array, d_array2);
        size *= 2;
    }

    size = 1024;
    std::cout << "    size\tHtD\tHtDPinned\tmalloc\tmallocHost\n";
    for (size_t i = 0; i < malloc_time.size(); i++) {
        std::cout << std::setprecision(1) << std::left << std::fixed
                  << std::setw(11) << size << "\t"
                  << malloc_HtD_time[i] << "\t" << mallochost_HtD_time[i]
                   << "\t" << malloc_time[i] << "\t" << mallochost_time[i]
                  << "\t" << dev2dev_time[i]
                  << "\t" << memset_time[i] << "\n";
        size *= 2;
    }
}

//nvcc -arch=sm_52 -std=c++11 -I../include Benchmarks.cu
//../src/Support/Host/PrintExt.cpp ../src/Support/Device/CudaUtil.cpp
