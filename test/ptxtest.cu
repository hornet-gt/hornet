#include <tuple>

template<typename... TArgs> struct DevTuple;

template<typename T>
struct DevTuple<T> {
    T value;

    __device__ __forceinline__
    DevTuple<T>& operator=(const DevTuple<T>& obj) {
        value = obj.value;
        return *this;
    }
};

template<typename T, typename... TArgs>
struct __align__(16) DevTuple<T, TArgs...> {
    DevTuple<TArgs...> tail;
    T value;


    __device__ __forceinline__
    DevTuple<T, TArgs...>& operator=(const DevTuple<T, TArgs...>& obj) {
        //value = obj.value;
        //tail  = obj.tail;
        *reinterpret_cast<int4*>(this) = reinterpret_cast<const int4&>(obj);
        return *this;
    }
};

template<typename... TArgs>
__host__ __device__ __forceinline__
DevTuple<TArgs...> make_devtuple(const TArgs&... args) {
    return DevTuple<TArgs...>{args...};
}


struct __align__(16) SS {
     int a;
     double b;
};

__device__ DevTuple<int, double> device_struct;

__global__ void ptxKernel() {
    //device_struct = {1, 3.0};//make_devtuple(1,3.0);
    device_struct = make_devtuple(3.0, 1);
}



int main() {
    ptxKernel<<<1, 1>>>();
}
