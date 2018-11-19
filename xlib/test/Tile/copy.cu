
template<typename T>
__global__
void copyKernel(const T* __restrict__ d_in, int size, T* __restrict__ d_out) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    d_in  += id;
    d_out += id;
    for (int i = 0; i < size; i += stride) {
        *d_out = *d_in;
        d_in  += stride;
        d_out += stride;
    }
}

int main(int argc, char* argv[]) {
    int size = 1024;
    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    copyKernel <<< 1, 128 >>> (d_in, size, d_out);
}
