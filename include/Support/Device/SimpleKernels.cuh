
template<typename T>
void __global__ memset_kernel(T* ptr, size_t num_items) {

}

template<typename T>
__global__
void memset_kernel(T* d_in_out, size_t num_items, T init_value) {
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO;

    const unsigned RATIO = sizeof(int4) > sizeof(T) ? sizeof(int4) / sizeof(T)
                                                    : 1;
    const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO;

    int         idx = blockIdx.x * blockDim.x + threadIdx.x;
    int      stride = gridDim.x * blockDim.x;
    int approx_size = xlib::lower_approx<WARP_SIZE>(num_items / THREAD_ITEMS);

    T storage[RATIO];
    #pragma unroll
    for (int K = 0; K < RATIO; K++)
        storage[K] = init_value;
    const auto& to_write = reinterpret_cast<int4&>(storage);

    auto d_tmp = reinterpret_cast<int4*>(d_in_out) + idx;
    for (int i = idx; i < approx_size; i += stride * THREAD_ITEMS) {
        #pragma unroll
        for (int K = 0; K < UNROLL_STEPS; K++)
            d_tmp[i + stride * K] = to_write;
    }
}
