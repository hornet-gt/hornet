
using hash_t = long long unsigned;
__device__ int d_count = 0;

const int HASH_FACTOR = 2;

__global__ void hashTableKernel(const edge2_t* __restrict__ d_edges,
                                hash_t* __restrict__ d_hash_space,
                                int batch_size, int V,
                                uint64_t random0, uint64_t random1,
                                edge2_t* __restrict__ d_out) {

    const uint64_t RANDOM[] = {16404704422080076895, 14912236932399409373};
    int     id = blockIdx.x * 256 + threadIdx.x;
    int stride = gridDim.x * 256;

    const unsigned    log_bins = xlib::ceil_log2(HASH_FACTOR * batch_size);
    const unsigned power2_bins = xlib::nearest_pow2_up(HASH_FACTOR * batch_size);
    const hash_t      NULL_KEY = static_cast<hash_t>(-1);

    int size = xlib::upper_approx_u<xlib::WARP_SIZE>(batch_size);
    for (int i = id; i < size; i += stride) {
        bool insert_flag = false;
        edge2_t item;

        if (i < batch_size) {
            item = d_edges[i];
            hash_t old_hash, hash = static_cast<uint64_t>(item.x) * V + item.y;
            uint64_t h1_value = xlib::multiplyShiftHash64(RANDOM[0], random0,
                                                          log_bins, hash);
            uint64_t h2_value = xlib::multiplyShiftHash64(RANDOM[1], random1,
                                                          log_bins, hash);
            int step = 0;
            do {
                unsigned index = (h1_value + step * h2_value)
                                 & (power2_bins - 1);
                old_hash = atomicCAS(d_hash_space + index, NULL_KEY, hash);

                if (old_hash == NULL_KEY) {
                    insert_flag = true;
                    break;
                }
                step++;
            } while ( old_hash != hash );
        }
        xlib::QueueWarp<xlib::cuQUEUE_MODE::BALLOT>
            ::store(item, insert_flag, d_out, &d_count);
    }
}


void hashTableTest(edge2_t* batch, int batch_size, int V, bool debug) {
    edge2_t* d_edges, *d_out;
    hash_t* d_hash_space;
    size_t hash_size = xlib::nearest_pow2_up(HASH_FACTOR * batch_size);
    SAFE_CALL( cudaMalloc(&d_edges, batch_size * sizeof(edge2_t)) );
    SAFE_CALL( cudaMalloc(&d_out, batch_size * sizeof(edge2_t)) );
    SAFE_CALL( cudaMalloc(&d_hash_space, hash_size * sizeof(hash_t)) );
    SAFE_CALL( cudaMemset(d_hash_space, 0xFF, hash_size * sizeof(hash_t)) );
    SAFE_CALL( cudaMemcpy(d_edges, batch, batch_size * sizeof(edge2_t),
                          cudaMemcpyHostToDevice) );

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    uint64_t max_random = 1 << (64 - xlib::ceil_log2(HASH_FACTOR * batch_size));
    std::uniform_int_distribution<edge_t> distribution(0, max_random - 1);
    uint64_t random0 = distribution(generator);
    uint64_t random1 = distribution(generator);
    //--------------------------------------------------------------------------
    timer2::Timer<timer2::DEVICE> TM;
    TM.start();

    hashTableKernel<<<xlib::uceil_div<256>(batch_size), 256>>>
        (d_edges, d_hash_space, batch_size, V, random0, random1, d_out);

    TM.stop();
    TM.print("HashTable");
    CHECK_CUDA_ERROR
    //--------------------------------------------------------------------------
    int unique_size;
    SAFE_CALL(cudaMemcpyFromSymbol(&unique_size, d_count, sizeof(int)));
    std::cout  << "   unique_size: " << unique_size << "\n\n";

    if (debug) {
        auto h_out = new edge2_t[batch_size];
        SAFE_CALL(cudaMemcpy(h_out, d_out, unique_size * sizeof(edge2_t),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < unique_size; i++)
            std::cout << h_out[i] << "\n";
        std::cout << std::endl;
        delete[] h_out;
    }
    SAFE_CALL( cudaFree(d_edges) );
    SAFE_CALL( cudaFree(d_out) );
    SAFE_CALL( cudaFree(d_hash_space) );
}
