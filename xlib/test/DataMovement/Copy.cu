#include <iostream>
#include <chrono>
#include <random>

#include <XLib.hpp>
using namespace xlib;

const int N = 1 << 25;

int main() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(
                                               std::numeric_limits<int>::min(),
                                               std::numeric_limits<int>::max());

    int* Input = new int[N];
    int* Output = new int[N];
    for (int i = 0; i < N; i++)
        Input[i] = distribution(generator);

    int *devInput, *devOutput;
    __SAFE_CALL( cudaMalloc(&devInput, N * sizeof(int)) )
    __SAFE_CALL( cudaMalloc(&devOutput, N * sizeof(int)) )
    __SAFE_CALL( cudaMemcpy(devInput, Input, N * sizeof(int),
                            cudaMemcpyHostToDevice) )

    const unsigned BLOCK_SIZE = 128;
    const unsigned grid_size = Div<BLOCK_SIZE>(N);

    /*xlib::copy<1><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);
    xlib::copy<2><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);
    xlib::copy<4><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);
    xlib::copy<8><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);
    xlib::copy<16><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);
    xlib::copy<32><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);
    xlib::copy<64><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);
    xlib::copy<128><<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);*/
    xlib::copy<<<grid_size, BLOCK_SIZE>>>(devInput, N, devOutput);

    __SAFE_CALL( cudaMemcpy(Output, devOutput, N * sizeof(int),
                            cudaMemcpyDeviceToHost) )

    for (int i = 0; i < N; i++) {
        if (Output[i] != Input[i]) {
            std::cout << "error" << std::endl;
            delete[] Input;
            delete[] Output;
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "correct" << std::endl;
    delete[] Input;
    delete[] Output;
}
