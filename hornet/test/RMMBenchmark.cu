#include <iomanip>
#include <iostream>
#include <vector>

#include <Host/Basic.hpp>//xlib::byte_t
#include <Host/PrintExt.hpp>//xlib::char_sequence and xlib::human_readable
#include <Device/Util/Timer.cuh>//timer::Timer

#include <StandardAPI.hpp>

using namespace hornets_nest;
using timer_duration_t = float;//return type of timer::Timer::duration()

int exec() {
#if defined(RMM_WRAPPER)
    constexpr size_t repeat_cnt = 10;
    size_t min_size = 1024;//1KB
    size_t round = 0;

    std::vector<timer_duration_t> v_alloc_time_host_cpp;//new and delete
    std::vector<timer_duration_t> v_alloc_time_host_cuda;//cudaMallocHost and cudaFreeHost
    std::vector<timer_duration_t> v_alloc_time_device_cuda;//cudaMalloc and cudaFree
    std::vector<timer_duration_t> v_alloc_time_device_rmm;//RMM_ALLOC and RMM_FREE

    timer::Timer<timer::DEVICE,timer::milli> my_timer;

    std::cout << "Computing (repeat count=" << repeat_cnt << ", RMM alloc mode=pool)" << std::endl;

    while (true) {
        size_t size = min_size << round;
        bool success = true;

        std::cout << "." << std::flush;

        {
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                std::unique_ptr<xlib::byte_t[]> h_p_cpp(new (std::nothrow) xlib::byte_t[size]);//no initialization, should not use std::make_unique here as this enforces initialization and is slower.
                if (h_p_cpp == nullptr) {
                    std::cout << std::endl;
                    std::cout << "new failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory." << std::endl;
                    success = false;
                    break;
                }
            }
            my_timer.stop();
            v_alloc_time_host_cpp.push_back(my_timer.duration());
        }

        if (success == true ) {
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                auto my_new = [](const size_t size) { xlib::byte_t* h_p_cuda; auto result = cudaMallocHost(&h_p_cuda, size); if (result == cudaSuccess) { return static_cast<xlib::byte_t*>(h_p_cuda); } else { return static_cast<xlib::byte_t*>(nullptr); } };
                auto my_del = [](xlib::byte_t* h_p_cuda) { SAFE_CALL(cudaFreeHost(h_p_cuda)); };
                std::unique_ptr<xlib::byte_t[], decltype(my_del)> h_p_cuda(my_new(size), my_del);
                if (h_p_cuda == nullptr) {
                    std::cout << std::endl;
                    std::cout << "cudaMallocHost failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory (that can be page-locked)." << std::endl;
                    success = false;
                    break;
                }
            }
            my_timer.stop();
            v_alloc_time_host_cuda.push_back(my_timer.duration());
        }

        if (success == true ) {
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                auto my_new = [](const size_t size) { xlib::byte_t* d_p_cuda; auto result = cudaMalloc(&d_p_cuda, size); if (result == cudaSuccess) { return static_cast<xlib::byte_t*>(d_p_cuda); } else { return static_cast<xlib::byte_t*>(nullptr); } };
                auto my_del = [](xlib::byte_t* d_p_cuda) { SAFE_CALL(cudaFree(d_p_cuda)); };
                std::unique_ptr<xlib::byte_t[], decltype(my_del)> d_p_cuda(my_new(size), my_del);
                if (d_p_cuda == nullptr) {
                    std::cout << std::endl;
                    std::cout << "cudaMalloc failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available device memory." << std::endl;
                    success = false;
                    break;
                }
            }
            my_timer.stop();
            v_alloc_time_device_cuda.push_back(my_timer.duration());
        }

        if (success == true ) {
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                auto my_new = [](const size_t size) { xlib::byte_t* d_p_rmm; auto result = RMM_ALLOC(&d_p_rmm, size, 0);/* by default, use the default stream, RMM_ALLOC instead of gpu::allocate to test return value, gpu::allocate calls std::exit on error */ if (result == RMM_SUCCESS) { return static_cast<xlib::byte_t*>(d_p_rmm); } else { return static_cast<xlib::byte_t*>(nullptr); } };
                auto my_del = [](xlib::byte_t* d_p_rmm) { gpu::free(d_p_rmm); };
                std::unique_ptr<xlib::byte_t[], decltype(my_del)> d_p_rmm(my_new(size), my_del);
                if (d_p_rmm == nullptr) {
                    std::cout << std::endl;
                    std::cout << "RMM_ALLOC failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available device memory (accessible to RMM)." << std::endl;
                    success = false;
                    break;
                }
            }
            my_timer.stop();
            v_alloc_time_device_rmm.push_back(my_timer.duration());
        }

        if (success == true ) {
            round++;
        }
        else {
            v_alloc_time_host_cpp.resize(round);
            v_alloc_time_host_cuda.resize(round);
            v_alloc_time_device_cuda.resize(round);
            v_alloc_time_device_rmm.resize(round);
            break;
        }
    }

    std::cout << "RESULT:" << std::endl;
    std::cout << std::setprecision(2) << std::right << std::fixed
              << std::setw(8)  << "SIZE"
              << std::setw(16) << "malloc"
              << std::setw(16) << "cudaMallocHost"
              << std::setw(16) << "cudaMalloc"
              << std::setw(16) << "rmmAlloc" << std::endl;
    xlib::char_sequence('-', 80);

    for (size_t i = 0; i < round; i++) {
        std::cout << std::setw(8)  << xlib::human_readable(min_size << i)
                  << std::setw(16) << v_alloc_time_host_cpp[i]
                  << std::setw(16) << v_alloc_time_host_cuda[i]
                  << std::setw(16) << v_alloc_time_device_cuda[i]
                  << std::setw(16) << v_alloc_time_device_rmm[i] << std::endl;
    }

    std::cout << "* unit: ms, measured time includes both memory allocation and deallocation." << std::endl;
#else
    std::cout << "RMM_WRAPPER should be defined to benchmark RMM." << std::endl;
#endif

    return 0;
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

