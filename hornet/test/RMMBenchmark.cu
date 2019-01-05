#include <iomanip>
#include <iostream>
#include <vector>

#include <Host/Basic.hpp>//xlib::byte_t
#include <Host/PrintExt.hpp>//xlib::char_sequence and xlib::human_readable
#include <Device/Util/Timer.cuh>//timer::Timer

#include <StandardAPI.hpp>

using namespace hornets_nest;
using timer_duration_t = float;//return type of timer::Timer::duration()

int main() {
#if defined(RMM_WRAPPER)
    constexpr size_t repeat_cnt = 10;
    size_t min_size = 1024;//1KB
    size_t round = 0;

    rmmOptions_t options;

    std::vector<timer_duration_t> v_alloc_time_host_cpp;//new and delete
    std::vector<timer_duration_t> v_alloc_time_host_cuda;//cudaMallocHost and cudaFreeHost
    std::vector<timer_duration_t> v_alloc_time_device_cuda;//cudaMalloc and cudaFree
    std::vector<timer_duration_t> v_alloc_time_device_rmm;//RMM_ALLOC and RMM_FREE

    timer::Timer<timer::DEVICE,timer::milli> my_timer;

    options.allocation_mode = PoolAllocation;
    //options.allocation_mode = CudaDefaultAllocation;
    options.initial_pool_size = 128 * 1024 * 1024;//128MB, relevant only if PoolAllocation is selected.

    if (options.allocation_mode == PoolAllocation) {
        std::cout << "Computing (repeat count=" << repeat_cnt << ", RMM alloc mode=pool, RMM init pool size=" << xlib::human_readable(options.initial_pool_size) << ")" << std::endl;
    }
    else {
        std::cout << "Computing (repeat count=" << repeat_cnt << ", RMM alloc mode=cuda default)" << std::endl;
    }

    my_timer.start();
    rmmInitialize(&options);
    my_timer.stop();
    auto rmm_initialize_duration = my_timer.duration();

    while (true) {
        size_t size = min_size << round;
        bool success = true;

        std::cout << "." << std::flush;

        {
            xlib::byte_t* h_p_cpp = nullptr;
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                h_p_cpp = new (std::nothrow) xlib::byte_t[size];
                if (h_p_cpp == nullptr) {
                    std::cout << std::endl;
                    std::cout << "new failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory." << std::endl;
                    success = false;
                    break;
                }
                delete[] h_p_cpp;
            }
            my_timer.stop();
            v_alloc_time_host_cpp.push_back(my_timer.duration());
        }

        if (success == true ) {
            xlib::byte_t* h_p_cuda = nullptr;
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                auto result = cudaSuccess;
                result = cudaMallocHost(&h_p_cuda, size);//cudaMallocHost instead of host::allocatePageLocked to test return value, host::allocatePageLocked calls std::exit on error.
                if (result != cudaSuccess) {
                    std::cout << std::endl;
                    std::cout << "cudaMallocHost failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory (that can be page-locked)." << std::endl;
                    success = false;
                    break;
                }
                result = cudaFreeHost(h_p_cuda);
                if (result != cudaSuccess) {
                    std::cout << std::endl;
                    std::cout << "cudaFreeHost failed (size=" << xlib::human_readable(size) << ")." << std::endl;
                    success = false;
                    break;
                }
            }
            my_timer.stop();
            v_alloc_time_host_cuda.push_back(my_timer.duration());
        }

        if (success == true ) {
            xlib::byte_t* d_p_cuda = nullptr;
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                auto result = cudaSuccess;
                result = cudaMalloc(&d_p_cuda, size);//directly invokes cudaMalloc (gpu::allocate invokes RMM_ALLOC if RMM_WRAPPER is defined).
                if (result != cudaSuccess) {
                    std::cout << std::endl;
                    std::cout << "cudaMalloc failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available device memory." << std::endl;
                    success = false;
                    break;
                }
                result = cudaFree(d_p_cuda);
                if (result != cudaSuccess) {
                    std::cout << std::endl;
                    std::cout << "cudaFree failed (size=" << xlib::human_readable(size) << ")." << std::endl;
                    success = false;
                    break;
                }
            }
            my_timer.stop();
            v_alloc_time_device_cuda.push_back(my_timer.duration());
        }

        if (success == true ) {
            xlib::byte_t* d_p_rmm = nullptr;
            my_timer.start();
            for (std::size_t i = 0; i < repeat_cnt; i++) {
                auto result = RMM_SUCCESS;
                result = RMM_ALLOC(&d_p_rmm, size, 0);//by default, use the default stream, RMM_ALLOC instead of gpu::allocate to test return value, gpu::allocate calls std::exit on error.
                if (result != RMM_SUCCESS) {
                    std::cout << std::endl;
                    std::cout << "RMM_ALLOC failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available device memory (accessible to RMM)." << std::endl;
                    success = false;
                    break;
                }
                result = RMM_FREE(d_p_rmm, 0);//by default, use the default stream
                if (result != RMM_SUCCESS) {
                    std::cout << std::endl;
                    std::cout << "RMM_FREE failed (size=" << xlib::human_readable(size) << ")." << std::endl;
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

    my_timer.start();
    rmmFinalize();
    my_timer.stop();
    auto rmm_finalize_duration = my_timer.duration();

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

    std::cout << "rmmInitialize:" << rmm_initialize_duration << " rmmFinalize:" << rmm_finalize_duration << std::endl;

    std::cout << "* unit: ms, measured time includes both memory allocation and deallocation." << std::endl;
#else
    std::cout << "RMM_WRAPPER should be defined to benchmark RMM." << std::endl;
#endif

    return 0;
}

