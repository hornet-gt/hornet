#include <iostream>

int main() {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    std::cout << devProp.major * 10 + devProp.minor
              << ";" << devProp.multiProcessorCount;
}
