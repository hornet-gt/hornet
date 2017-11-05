#include <XLib.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <cub/cub.cuh>

using namespace xlib;
using namespace timer;

struct Struct {
    int a;
    int b;
    __host__ __device__ __forceinline__
    Struct() {}

    __host__ __device__ __forceinline__
    Struct(int a, int b) : a(a), b(b) {}
};



int main(int argc, char* argv[]) {
    Timer<DEVICE> TM;

    //--------------------------------------------------------------------------
    //   HOST
    /*using T = Struct;
    using R = int;
    const int size = 32 * 4 + 3;// 1 << 27;

    auto h_array = new T[size];
    std::fill(h_array, h_array + size, Struct(2,0));
    h_array[100] = Struct(4, 0);

    h_array[120] = Struct(7, 0);

    T* d_array;
    R* d_out;
    cuMalloc(d_array, size);
    cuMalloc(d_out, 1);
    cuMemcpyToDevice(h_array, size, d_array);
    cuMemcpyToDevice(R(0), d_out);
    //--------------------------------------------------------------------------
    const auto& thread_op = [] __device__ (const T& a, const T& b) {
                                return Struct(::max(a.a, b.a), 0);
                            };
    const auto&   warp_op = [] __device__ (const T& value, R* __restrict__ d_out) {
                                    xlib::WarpReduce<>::atomicMax(value.a, d_out);
                            };

    TM.start();

    const auto& select_op = [] __device__ (const Struct& value) {
                                return value.a;
                            };
    auto result = xlib::device_reduce::reduce_argmax<1, int>(d_array, size, select_op);
    std::cout << result.first << " " << result.second << std::endl;

    TM.stop();
    TM.print("reduce 1");*/

//--------------------------------------------------------------------------
    using T = int;
    const int size = 1 << 29;

    auto h_array = new T[size];
    std::fill(h_array, h_array + size, 1);

    T* d_array, *d_out;
    cuMalloc(d_array, size);
    cuMalloc(d_out, 1);
    cuMemcpyToDevice(h_array, size, d_array);
    cuMemcpyToDevice(0, d_out);
    //--------------------------------------------------------------------------
    TM.start();

    xlib::device_reduce::add<1>(d_array, size, d_out);

    TM.stop();
    TM.print("reduce 1");

    //--------------------------------------------------------------------------
    TM.start();

    xlib::device_reduce::add<2>(d_array, size, d_out);

    TM.stop();
    TM.print("reduce 2");

    //--------------------------------------------------------------------------
    TM.start();

    xlib::device_reduce::add<4>(d_array, size, d_out);

    TM.stop();
    TM.print("reduce 4");

    //--------------------------------------------------------------------------
    TM.start();

    xlib::device_reduce::add<8>(d_array, size);

    TM.stop();
    TM.print("reduce 8");

    //--------------------------------------------------------------------------
    TM.start();

    xlib::device_reduce::add<16>(d_array, size);

    TM.stop();
    TM.print("reduce 16");

    //--------------------------------------------------------------------------
    TM.start();

    xlib::device_reduce::add<32>(d_array, size);

    TM.stop();
    TM.print("reduce 32");
    //--------------------------------------------------------------------------
    TM.start();

    xlib::device_reduce::add<64>(d_array, size);

    TM.stop();
    TM.print("reduce 64");

    //--------------------------------------------------------------------------

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_array, d_out,
                           size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    TM.start();

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_array, d_out,
                           size);

    TM.stop();
    TM.print("cub");
    //std::cout << size << std::endl
    //          << result << std::endl;*/
}
