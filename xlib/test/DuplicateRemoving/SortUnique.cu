
__global__ void mergeKernel(const edge_t* __restrict__ d_edges1,
                            const edge_t* __restrict__ d_edges2,
                            edge2_t* __restrict__ d_edges,
                            int batch_size) {

    int     id = blockIdx.x * 256 + threadIdx.x;
    int stride = gridDim.x * 256;
    for (int i = id; i < batch_size; i += stride)
        d_edges[i] = xlib::make2<edge_t>(d_edges1[i], d_edges2[i]);
}

void sortUniqueTest(edge2_t* batch, int batch_size, int V, bool debug) {
    auto h_edges1 = new edge_t[batch_size];
    auto h_edges2 = new edge_t[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_edges1[i] = batch[i].x;
        h_edges2[i] = batch[i].y;
    }

    edge2_t* d_in, *d_out;
    edge_t* d_edges1, *d_edges2, *d_edges1_out, *d_edges2_out;
    int* d_unique_egdes;
    SAFE_CALL( cudaMalloc(&d_in, batch_size * sizeof(edge2_t)) );
    SAFE_CALL( cudaMalloc(&d_out, batch_size * sizeof(edge2_t)) );
    SAFE_CALL( cudaMalloc(&d_unique_egdes, sizeof(int)) );
    SAFE_CALL( cudaMalloc(&d_edges1, batch_size * sizeof(edge_t)) );
    SAFE_CALL( cudaMalloc(&d_edges2, batch_size * sizeof(edge_t)) );
    SAFE_CALL( cudaMalloc(&d_edges1_out, batch_size * sizeof(edge_t)) );
    SAFE_CALL( cudaMalloc(&d_edges2_out, batch_size * sizeof(edge_t)) );
    SAFE_CALL( cudaMemcpy(d_edges1, h_edges1, batch_size * sizeof(edge_t),
                          cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(d_edges2, h_edges2, batch_size * sizeof(edge_t),
                          cudaMemcpyHostToDevice) );

    void*      d_temp_storage1 = nullptr;
    size_t temp_storage_bytes1 = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage1, temp_storage_bytes1,
                                    d_edges2, d_edges2_out,
                                    d_edges1, d_edges1_out,
                                    batch_size, 0, xlib::ceil_log2(V));
    cudaMalloc(&d_temp_storage1, temp_storage_bytes1);

    void*  d_temp_storage3     = nullptr;
    size_t temp_storage_bytes3 = 0;
    cub::DeviceSelect::Unique(d_temp_storage3, temp_storage_bytes3,
                              d_in, d_out, d_unique_egdes, batch_size);
    cudaMalloc(&d_temp_storage3, temp_storage_bytes3);
    //--------------------------------------------------------------------------
    timer2::Timer<timer2::DEVICE> TM;
    TM.start();

    cub::DeviceRadixSort::SortPairs(d_temp_storage1, temp_storage_bytes1,
                                    d_edges2, d_edges2_out,
                                    d_edges1, d_edges1_out,
                                    batch_size, 0, xlib::ceil_log2(V));

    cub::DeviceRadixSort::SortPairs(d_temp_storage1, temp_storage_bytes1,
                                    d_edges1_out, d_edges1,
                                    d_edges2_out, d_edges2,
                                    batch_size, 0, xlib::ceil_log2(V));

    mergeKernel<<<xlib::uceil_div<256>(batch_size), 256>>>
        (d_edges1, d_edges2, d_in, batch_size);

    cub::DeviceSelect::Unique(d_temp_storage3, temp_storage_bytes3,
                              d_in, d_out, d_unique_egdes, batch_size);
    TM.stop();
    TM.print("SortUnique");
    CHECK_CUDA_ERROR
    //--------------------------------------------------------------------------
    int unique_size;
    SAFE_CALL(cudaMemcpy(&unique_size, d_unique_egdes, sizeof(int),
                         cudaMemcpyDeviceToHost));
        std::cout  << "   unique_size: " << unique_size << "\n\n";

    if (debug) {
        auto h_edges = new edge2_t[batch_size];
        SAFE_CALL(cudaMemcpy(h_edges, d_in, unique_size * sizeof(edge2_t),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < unique_size; i++)
            std::cout << h_edges[i] << "\n";
        std::cout << std::endl;
        delete[] h_edges;
    }
    delete[] h_edges1;
    delete[] h_edges2;
    SAFE_CALL( cudaFree(d_unique_egdes) );
    SAFE_CALL( cudaFree(d_in) );
    SAFE_CALL( cudaFree(d_out) );
    SAFE_CALL( cudaFree(d_edges1) );
    SAFE_CALL( cudaFree(d_edges2) );
    SAFE_CALL( cudaFree(d_edges1_out) );
    SAFE_CALL( cudaFree(d_edges2_out) );
    SAFE_CALL( cudaFree(d_temp_storage1) );
    SAFE_CALL( cudaFree(d_temp_storage3) );
}
