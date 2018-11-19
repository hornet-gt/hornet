// Example program
#include <iostream>
#include <iterator>
#include "XLib.hpp"
using namespace xlib;
using namespace graph;
//------------------------------------------------------------------------------

__device__ int F_array[64];

__device__ int Ouptput[64];

__global__ void prova1() {
    cu_array<int, 32> A = cu_array<int, 32>(F_array, 64);

    for (auto& it : A)
        Ouptput[threadIdx.x] = it;
        //printf("threadIdx.x %d \t %d\n", threadIdx.x, it);
    //printf("threadIdx.x %d \t %d\n", threadIdx.x, A.begin().val);
}

__global__ void prova2() {
    const int globalID = 32 * blockIdx.x + threadIdx.x;
    for (int i = globalID; i < 64; i += gridDim.x * 32)
        Ouptput[threadIdx.x] = F_array[i];
}


//------------------------------------------------------------------------------

__device__ int2 Adj_array[5];


__global__ void prova3() {
    //auto A = NQfrontier<32>(F_array, 5, Adj_array);

    //for (auto it : A)
        //Ouptput[threadIdx.x] = it.start;
    //    printf("threadIdx.x %d \t %d\n", threadIdx.x, it.end);
    //printf("threadIdx.x %d \t %d\n", threadIdx.x, (*A.begin()).start);
}

__global__ void provaB() {
    auto node = example_iterator_node(F_array, Adj_array);
    auto A = frontier<32, example_iterator_node>(5, node);

    for (auto it : A)
        Ouptput[threadIdx.x] = it.start;
        //printf("threadIdx.x %d \t %d\n", threadIdx.x, it.end);
    //printf("threadIdx.x %d \t %d\n", threadIdx.x, (*A.begin()).start);
}


__global__ void prova4() {
    const int globalID = 32 * blockIdx.x + threadIdx.x;
    const int size = 5;
    const int max_size = xlib::upperApprox<32>(size);
    for (int i = globalID; i < max_size; i += gridDim.x * 32) {
        cusize_t start, end, degree;
        if (i < size) {
            int index = F_array[i];
            cusize2_t adj = Adj_array[index];
            start = adj.x;
            end = adj.y;
            degree = end - start;
        } else {
            end = xlib::numeric_limits<cusize_t>::min;
            degree = 0;
        }
        Ouptput[threadIdx.x] = start;
        Ouptput[threadIdx.x] = degree;
    }
}


__device__ __forceinline__  void ff(int a ) {

}





__device__ auto fun = [=](int i) -> void {
                printf("%d\n", i);
            };

template<typename FUNa>
__device__ __forceinline__ void exec(FUNa fun, int v) {
    fun(v);
}

template<decltype(fun) FUN>
__device__ __forceinline__ void exec2(int v) {
    FUN(v);
}

__global__ void provaLambda() {
    int v = 3;

    /*auto fun = [=](int i) -> void {
                    printf("%d\n", i);
                };*/
    exec(fun, v);
    exec2<fun>(v);
}

int main() {
    int input[64];
    for (int i = 0; i < 64; i++)
        input[i] = i;
    cudaMemcpyToSymbol(F_array, input, 64 * sizeof(int));

    int2 adj_input[64];
    adj_input[0] = make_int2(0, 3);
    adj_input[1] = make_int2(3, 7);
    adj_input[2] = make_int2(7, 10);
    adj_input[3] = make_int2(10, 12);
    adj_input[4] = make_int2(12, 15);
    cudaMemcpyToSymbol(Adj_array, adj_input, 5 * sizeof(int2));

    prova3<<<1, 32>>>();
    provaB<<<1, 32>>>();
    prova4<<<1, 32>>>();
        //prova2<<<1, 32>>>();
    CUDA_ERROR("prova");
}
