#include <XLib.hpp>

__device__ float array[5];

__global__ void prova2() {
    float val = threadIdx.x;
    xlib::WarpReduce<>::AtomicMin(val, array + 3);
}

int main() {
    prova2<<<1, 32>>>();
}
