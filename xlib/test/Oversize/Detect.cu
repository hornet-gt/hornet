
__device__ int d_vertex_count = 0;
__device__ int  d_egdes_count = 0;

__global__ void histogramKernel(const edge2_t* __restrict__ d_batch,
                                int* __restrict__ d_histogram,
                                int* __restrict__ d_aux,
                                int batch_size) {

    int     id = blockIdx.x * 256 + threadIdx.x;
    int stride = gridDim.x * 256;
    for (int i = id; i < batch_size; i += stride)
        d_aux[i] = atomicAdd(d_histogram + d_batch[i].x, 1);
}

__global__ void detectKernel(const edge2_t* __restrict__ d_batch,
                             int* __restrict__ d_histogram,
                             int* __restrict__ d_aux,
                             edge_t*  __restrict__ d_ovesize_vertices,
                             edge2_t* __restrict__ d_ovesize_edges,
                             edge_t*  __restrict__ d_edge_block,
                             int batch_size, int threshold) {

    int     id = blockIdx.x * 256 + threadIdx.x;
    int stride = gridDim.x * 256;
    int size = xlib::upper_approx_u<xlib::WARP_SIZE>(batch_size);
    const int REGS = 16;
    edge2_t edge_queue[REGS/2];
    edge_t vertex_queue[REGS];
    int edge_count = 0, vertex_count = 0;

    for (int i = id; i < size; i += stride) {
        if (i < batch_size) {
            edge2_t item = d_batch[i];
            if (d_histogram[item.x] > threshold) {
                //printf("%d\t%d\n", item.x, item.y);
                edge_queue[edge_count++] = item;
                if (d_aux[i] == 0)
                    vertex_queue[vertex_count++] = item.x;
            }
            else
                d_edge_block[d_aux[i]] = item.x;
        }
        if (__any(edge_count == REGS/2)) {
            xlib::QueueWarp<xlib::cuQUEUE_MODE::SIMPLE>
                    ::store(edge_queue, edge_count,
                            d_ovesize_edges, &d_egdes_count);
            edge_count = 0;
        }
        if (__any(vertex_count == REGS)) {
            xlib::QueueWarp<xlib::cuQUEUE_MODE::SIMPLE>
                    ::store(vertex_queue, vertex_count,
                            d_ovesize_vertices, &d_vertex_count);
            vertex_count = 0;
        }
    }
    xlib::QueueWarp<xlib::cuQUEUE_MODE::SIMPLE>
            ::store(edge_queue, edge_count, d_ovesize_edges, &d_egdes_count);
    xlib::QueueWarp<xlib::cuQUEUE_MODE::SIMPLE>
            ::store(vertex_queue, vertex_count,
                    d_ovesize_vertices, &d_vertex_count);
}

void detectTest(edge2_t* batch, int batch_size, int V, int threshold,
                bool debug) {
    edge2_t* d_batch, *d_ovesize_edges;
    int* d_histogram, *d_aux;
    edge_t* d_ovesize_vertices, *d_edge_block;
    SAFE_CALL( cudaMalloc(&d_batch, batch_size * sizeof(edge2_t)) );
    SAFE_CALL( cudaMalloc(&d_histogram, V * sizeof(int)) );
    SAFE_CALL( cudaMalloc(&d_ovesize_vertices, batch_size * sizeof(edge_t)) );
    SAFE_CALL( cudaMalloc(&d_ovesize_edges, batch_size * sizeof(edge2_t)) );
    SAFE_CALL( cudaMalloc(&d_edge_block, batch_size * sizeof(edge_t)) );
    SAFE_CALL( cudaMalloc(&d_aux, batch_size * sizeof(int)) );
    SAFE_CALL( cudaMemset(d_histogram, 0x0, V * sizeof(int)) );
    SAFE_CALL( cudaMemset(d_aux, 0x0, batch_size * sizeof(int)) );
    SAFE_CALL( cudaMemcpy(d_batch, batch, batch_size * sizeof(edge2_t),
                          cudaMemcpyHostToDevice) );
    //--------------------------------------------------------------------------
    timer2::Timer<timer2::DEVICE> TM;
    TM.start();

    histogramKernel<<<xlib::uceil_div<256>(batch_size), 256>>>
        (d_batch, d_histogram, d_aux, batch_size);

    detectKernel<<<xlib::uceil_div<256>(batch_size), 256>>>
            (d_batch, d_histogram, d_aux, d_ovesize_vertices,
             d_ovesize_edges, d_edge_block, batch_size, threshold);

    TM.stop();
    TM.print("DetectKernel");
    CHECK_CUDA_ERROR
    //--------------------------------------------------------------------------
    int h_oversize;
    SAFE_CALL(cudaMemcpyFromSymbol(&h_oversize, d_vertex_count, sizeof(int)));
    std::cout  << "  n. of oversize: " << h_oversize << "\n\n";

    if (debug) {
        auto h_histogram = new int[V];
        SAFE_CALL(cudaMemcpy(h_histogram, d_histogram, V * sizeof(int),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < V; i++)
            std::cout << h_histogram[i] << "\n";
        std::cout << std::endl;
        delete[] h_histogram;

        auto h_aux = new int[batch_size];
        SAFE_CALL(cudaMemcpy(h_aux, d_aux, batch_size * sizeof(int),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < batch_size; i++)
            std::cout << h_aux[i] << "\n";
        std::cout << std::endl;
        delete[] h_aux;

        /*auto h_aux = new int[batch_size];
        SAFE_CALL(cudaMemcpy(h_aux, d_aux, batch_size * sizeof(int),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < batch_size; i++)
            std::cout << h_aux[i] << "\n";
        std::cout << std::endl;
        delete[] h_aux;*/

        auto h_out = new edge_t[h_oversize];
        SAFE_CALL(cudaMemcpy(h_out, d_ovesize_vertices,
                             h_oversize * sizeof(edge_t),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < h_oversize; i++)
            std::cout << h_out[i] << "\n";
        std::cout << std::endl;
        delete[] h_out;
    }
    SAFE_CALL( cudaFree(d_batch) );
    SAFE_CALL( cudaFree(d_histogram) );
    SAFE_CALL( cudaFree(d_ovesize_vertices) );
    SAFE_CALL( cudaFree(d_aux) );
}
