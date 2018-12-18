#include "Core/DataLayout/DataLayout.cuh"
#include "Core/DataLayout/DataLayoutDev.cuh"

using namespace hornet;
/*
template<typename... TArgs>
SoA<TArgs...>& SoA<TArgs...>::operator=(const AoSData<TArgs...>& obj) {
    RecursiveAssign<0, NUM_ARGS - 1, TArgs...>::apply(obj, _pitch, _ptr);
    return *this;
}*/

//==============================================================================

__device__ AoSData<int, int> dev_ll;

__global__ void ptxKernel2() {
    dev_ll = AoSData<int, int>(1,2);
}

int main() {
    int   A[] = { 1, 2, 3 };
    float B[] = { 1.0, 2.3, 3.0 };

    void* array[2] = { A, B };
    AoS<int, float> aos_struct(array, 3);

    SoA<int, float> soa_struct(array, 3);

    std::cout << aos_struct[1].get<0>() << std::endl;
    std::cout << aos_struct[1].get<1>() << std::endl << std::endl;

    std::cout << aos_struct[1] << std::endl;
    std::cout << soa_struct[1] << std::endl;

    AoSData<int, float> b = soa_struct[1];
    std::cout << soa_struct[1] << std::endl;
    soa_struct[1] = AoSData<int, float>(5, 6.0);
    std::cout << soa_struct[1] << std::endl;

    ptxKernel2<<<1,1>>>();

    BestLayout<int, double> a(array, 3);
}
