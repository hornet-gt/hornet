#include <iostream>

int main() {
    int num_gpus;
    if (cudaGetDeviceCount(&num_gpus) != cudaSuccess)
        return EXIT_FAILURE;

    std::cout << num_gpus << ";";
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp devive_prop;
        if (cudaGetDeviceProperties(&devive_prop, i) != cudaSuccess)
            return EXIT_FAILURE;
        std::cout << devive_prop.major * 10 + devive_prop.minor << ";"
                  << devive_prop.multiProcessorCount            << ";"
                  << devive_prop.name;
        if (i < num_gpus - 1)
            std::cout << ";";
    }
}
