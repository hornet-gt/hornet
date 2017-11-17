#include <iostream>

int main() {
    cudaDeviceProp devive_prop;
    cudaGetDeviceProperties(&devive_prop, 0);
    std::cout << devive_prop.major * 10 + devive_prop.minor
              << ";" << devive_prop.multiProcessorCount
              << ";" << devive_prop.name;
}
