#include <StandardAPI.hpp>
#include <Device/Primitives/SimpleKernels.cuh>
#include <Device/Util/Timer.cuh>
#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace timer;
using namespace hornets_nest;
using xlib::byte_t;
using ttime_t = float;

int exec(void) {
    size_t size = 1024;
    Timer<DEVICE> TM;

    std::vector<ttime_t> allocation_time;
    std::vector<ttime_t> allocation_pinned_time;
    std::vector<ttime_t> H2D_time;
    std::vector<ttime_t> H2D_pinned_time;
    std::vector<ttime_t> D2D_time;
    std::vector<ttime_t> memset_time;
    byte_t* d_array, *h_array_pinned;

    std::cout << "Computing";

    while (true) {
        std::cout << "." << std::flush;
        //======================================================================
        TM.start();

        if (cudaMalloc(&d_array, size) != cudaSuccess)//cudaMalloc instead of gpu::allocate to test return value, gpu::allocate calls std::exit() on error.
            break;

        TM.stop();
        allocation_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        auto h_array = new byte_t[size];
        TM.start();

        host::copyToDevice(h_array, size, d_array);

        TM.stop();
        delete[] h_array;
        H2D_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        TM.start();

        host::allocatePageLocked(h_array_pinned, size);

        TM.stop();

        allocation_pinned_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        TM.start();

        host::copyToDeviceAsync(h_array_pinned, size, d_array);

        TM.stop();
        host::freePageLocked(h_array_pinned);
        H2D_pinned_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        TM.start();

        gpu::memsetZero(d_array, size);

        TM.stop();
        memset_time.push_back(TM.duration());
        //----------------------------------------------------------------------
        byte_t* d_array2;
        if (cudaMalloc(&d_array2, size) == cudaSuccess) {//cudaMalloc instead of gpu::allocate to test return value, gpu::allocate calls std::exit() on error.
            TM.start();

            gpu::copyToDevice(d_array, size, d_array2);

            TM.stop();
            D2D_time.push_back(TM.duration());
            SAFE_CALL(cudaFree(d_array2));
        }
        else {
            D2D_time.push_back(std::nan(""));
        }
        SAFE_CALL(cudaFree(d_array));
        //----------------------------------------------------------------------
        size *= 2;
    }
    size = 1024;
    std::cout << "\n\n" << std::setprecision(2) << std::right << std::fixed
              << std::setw(8)  << "SIZE"
              << std::setw(12) << "cudaMalloc"
              << std::setw(18) << "cudaMallocPinned"
              << std::setw(11) << "MemcpyHtD"
              << std::setw(14) << "MemcpyHtDPin"
              << std::setw(11) << "MemcpyDtD"
              << std::setw(8)  << "Memset" << std::endl;
    xlib::char_sequence('-', 80);

    for (size_t i = 0; i < H2D_time.size(); i++) {
        std::cout << std::setw(8)  << xlib::human_readable(size)
                  << std::setw(12) << allocation_time[i]
                  << std::setw(18) << allocation_pinned_time[i]
                  << std::setw(11) << H2D_time[i]
                  << std::setw(14) << H2D_pinned_time[i]
                  << std::setw(11) << D2D_time[i]
                  << std::setw(8)  << memset_time[i] << "\n";
        size *= 2;
    }
    //==========================================================================
    Timer<DEVICE> TM2(2);

    xlib::byte_t array[4 * xlib::MB];
    gpu::allocate(d_array, 4 * xlib::MB);
    host::allocatePageLocked(h_array_pinned, 4 * xlib::MB);

    TM2.start();

    gpu::copyToHost(d_array, 4 * xlib::MB, array);

    TM2.stop();
    TM2.print("Stack");

    TM2.start();

    gpu::copyToHostAsync(d_array, 4 * xlib::MB, h_array_pinned);

    TM2.stop();
    TM2.print("Pinned");

    gpu::free(d_array);
    host::freePageLocked(h_array_pinned);

    return 0;
}

int main(void) {
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

