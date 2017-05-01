
/////////////////
/// C++11 API ///
/////////////////

template<typename Lambda>
__global__ void forAllKernel(int size, Lambda lambda) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < size; i += stride)
        lambda(i);
}

template<typename Lambda>
void forAll(size_t size, Lambda lambda) {
    forAllKernel<<< xlib::ceil_div<BLOCK_SIZE_OP>(size), BLOCK_SIZE_OP >>>
        (size, lambda);
}

template<typename Lambda>
void forAllnumV(Lambda lambda) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nV, num_items);
    forAll(num_items, lambda);
}

template<typename Lambda>
void forAllnumE(Lambda lambda) {
    size_t num_items;
    cuMemcpyFromSymbol(d_nE, num_items);
    forAll(num_items, lambda);
}

//------------------------------------------------------------------------------

template<typename Lambda>
void forAllVertices(Lambda lambda) {

}

template<typename Lambda>
void forAllEdges(Lambda lambda) {

}
