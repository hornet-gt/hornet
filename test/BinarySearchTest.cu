#include "Host/Numeric.hpp"
#include "Device/Util/DeviceProperties.cuh"
#include "Device/Util/PrintExt.cuh"
#include "Device/Util/Algorithm.cuh"
#include "Device/Primitives/BinarySearchLB.cuh"
#include "Device/Primitives/impl/BinarySearchLB2.i.cuh"
#include "Device/Primitives/MergePathLB.cuh"
#include "Device/Util/Timer.cuh"
//#include <Graph/GraphBase.hpp>
#include <Graph/GraphStd.hpp>
#include <Graph/GraphWeight.hpp>
#include <Graph/BellmanFord.hpp>
#include <Graph/Dijkstra.hpp>

#include <iostream>
#include "Device/Util/SafeCudaAPIAsync.cuh" //cuMemset0x00Async

#include "Device/Util/Timer.cuh"
#include "Device/DataMovement/impl/Block.i.cuh"
#include <cooperative_groups.h>
//#define ENABLE_MGPU
#include <random>
#include <chrono>

#if defined(ENABLE_MGPU)
    #include <moderngpu/kernel_load_balance.hxx>
#endif

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void MergePathTest2(const int* __restrict__ d_partitions,
                    int                     num_partitions,
                    const int* __restrict__ d_prefixsum,
                    int                     prefixsum_size,
                    int* __restrict__       d_pos,
                    int* __restrict__       d_offset) {
    __shared__ int smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&](int pos, int, int index) {
                             d_pos[index] = pos;
                             //d_offset[index] = offset;
                        };
    //xlib::binarySearchLB2<BLOCK_SIZE, ITEMS_PER_BLOCK / BLOCK_SIZE, true>
    //    (d_partitions, num_partitions, d_prefixsum, prefixsum_size, smem, lambda);

    xlib::mergePathLB<BLOCK_SIZE, ITEMS_PER_BLOCK>
        (d_partitions, num_partitions, d_prefixsum, prefixsum_size, smem, lambda);
}

using namespace xlib;
using namespace graph;
using namespace timer;

const bool PRINT      = false;
const int  BLOCK_SIZE = 128;


__device__ int d_value;

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void copyKernel(const int* __restrict__ input, int num_blocks, int smem_size) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    __shared__ int smem[ITEMS_PER_BLOCK];

    for (int i = blockIdx.x; i < num_blocks; i += gridDim.x) {
        block::StrideOp<0, ITEMS_PER_BLOCK, BLOCK_SIZE>
            ::copy(input + i * ITEMS_PER_BLOCK, smem_size, smem);
        /*auto smem_tmp = smem + threadIdx.x;
        auto d_tmp    = input + i * ITEMS_PER_BLOCK + threadIdx.x;

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_BLOCK; i += BLOCK_SIZE)
            smem_tmp[i] = (i + threadIdx.x < smem_size) ? d_tmp[i] : 0;*/

        if (threadIdx.x > 1023)
            d_value = smem[threadIdx.x];
    }
}


template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void copyKernel2(const int* __restrict__ input, int num_blocks, int smem_size) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //__shared__ int smem[ITEMS_PER_BLOCK];

    for (int i = blockIdx.x; i < num_blocks; i += gridDim.x) {
        auto smem_tmp = xlib::dyn_smem + threadIdx.x;
        auto d_tmp    = input + i * ITEMS_PER_BLOCK + threadIdx.x;

        for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE) {
            *smem_tmp = *d_tmp;
            smem_tmp += BLOCK_SIZE;
            d_tmp    += BLOCK_SIZE;
        }

        if (threadIdx.x > 1023)
            d_value = xlib::dyn_smem[threadIdx.x];
    }
}


__global__
void noLambdaKernel(const int* __restrict__ ptr2, int* __restrict__ ptr1, int size) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride) {
        ptr1[i] = ptr2[i];
        ptr1[i + 10] = ptr2[i + 10];
        ptr1[i + 20] = ptr2[i + 20];
    }
}

template<typename Lambda>
__global__
void lambdaKernel(Lambda lambda, int size) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride)
        lambda(i);
}


template<typename Lambda, typename... TArgs>
__global__
void lambdaKernel2(Lambda lambda, int size, TArgs* __restrict__ ... args) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride)
        lambda(i, args...);
}

struct LL {
    int*       __restrict__ ptr1;
    const int* __restrict__ ptr2;

    __device__ __forceinline__
    void operator()(int i) {
        const int* __restrict__ vv2 = ptr2;
        int*       __restrict__ vv1 = ptr1;

        vv1[i] = vv2[i];
        vv1[i + 10] = vv2[i + 10];
        vv1[i + 20] = vv2[i + 20];
    }
};

int main(int argc, char* argv[]) {
    //xlib::NaturalIterator natural_iterator;
    /*const int NUM_THREADS = 5;
    int  offsets[] = { 0, 2, 2, 4, 8, 16 };
    int*   offset2 = offsets + 1;
    const int SIZE = sizeof(offsets) / sizeof(int) - 1;
    int  max_value = offsets[SIZE];
    int  num_merge = max_value + SIZE;
    int      ITEMS = xlib::ceil_div<NUM_THREADS>(num_merge);

    std::cout << std::endl << "ITEMS  " <<  ITEMS << std::endl;
    std::cout << "........................" << std::endl;
    for (int i = 0; i < NUM_THREADS; i++) {
        auto range_start = merge_path_search(offset2, SIZE,
                                             natural_iterator, max_value,
                                             min(i * ITEMS, num_merge));
        auto   range_end = merge_path_search(offset2, SIZE,
                                             natural_iterator, max_value,
                                             min((i + 1) * ITEMS, num_merge));
        std::cout << range_start.x << "  " << range_end.x << "\t\t"
                  << range_start.y << "  " << range_end.y
                  << "\t\t"
                  << offset2[range_start.x] << "\t\t"
                  << (range_end.x - range_start.x) + (range_end.y - range_start.y)
                  << std::endl;

        for (int j = 0; j < ITEMS; j++) {
            if (range_start.y < offset2[range_start.x]) {
                std::cout << "       " << range_start.y << "\n";
                range_start.y++;
            }
            else {
                //std::cout << "       " << range_start.x << "\n";
                range_start.x++;
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return 0;*/
    using namespace graph;
    GraphStd<int, int> graph1;
    graph1.read(argv[1]);

    graph1.print_degree_distrib();
    graph1.print_analysis();

    auto weights = new int[graph1.nV()];
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                .count();
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> distrib(0, 100);
    std::generate(weights, weights + graph1.nV(),
                  [&](){ return distrib(engine); } );

    GraphWeight<int, int, int> graph_weight(graph1.csr_out_edges(), graph1.nV(),
                                            graph1.csr_out_edges(), graph1.nE(),
                                            weights);


    Timer<HOST> TM1;
    /*BellmanFord<int, int, int> bellman_ford(graph_weight);

    TM1.start();

    for (int i = 0; i < graph1.nV(); i++) {
        bellman_ford.run(i);
        bellman_ford.reset();
    }
    TM1.stop();
    TM1.print("BellmanFord");*/

    Dijkstra<int, int, int> dijkstra(graph_weight);

    TM1.start();

    for (int i = 0; i < graph1.nV(); i++) {
        dijkstra.run(i);
        dijkstra.reset();
    }
    TM1.stop();
    TM1.print("Dijkstra");
    return 1;



    const int THREAD_ITEMS    = 11;
    const int ITEMS_PER_BLOCK = BLOCK_SIZE * THREAD_ITEMS;

    int num_blocks_copy = 100000;

    int* d_input;
    cuMalloc(d_input, ITEMS_PER_BLOCK * num_blocks_copy);

    Timer<DEVICE, micro> TM;
    TM.start();

    copyKernel<ITEMS_PER_BLOCK, BLOCK_SIZE>
        <<< num_blocks_copy, BLOCK_SIZE >>> (d_input, num_blocks_copy, 9 * BLOCK_SIZE);

    TM.stop();
    TM.print("copy1");
    TM.start();

    copyKernel2<ITEMS_PER_BLOCK, BLOCK_SIZE>
        <<< num_blocks_copy, BLOCK_SIZE >>> (d_input, num_blocks_copy, 9 * BLOCK_SIZE);

    TM.stop();
    TM.print("copy2");

    return 1;
    /*int* __restrict__ ptr1, * __restrict__ ptr2;
    lambdaKernel<<<1,1>>>( [=]__device__(int i){
                            ptr1[i] = ptr2[i];
                            ptr1[i + 10] = ptr2[i + 10];
                            ptr1[i + 20] = ptr2[i + 20];
                          }, 10);

    lambdaKernel<<<1,1>>>( LL{ptr1, ptr2}, 10);

    lambdaKernel2<<<1,1>>>([]__device__(int i, const int* ptr2, int* ptr1){
                            ptr1[i] = ptr2[i];
                            ptr1[i + 10] = ptr2[i + 10];
                            ptr1[i + 20] = ptr2[i + 20];
                          }, 10, ptr2, ptr1);

    noLambdaKernel<<<1,1>>>( ptr2, ptr1, 10);*/



    /*TM.start();

    xlib::global_sync_reset();

    TM.stop();
    TM.print("global_reset");
    TM.start();

    globalSyncTest2<64> <<< 56 * 2048 / 256, 256 >>> ();

    TM.stop();
    TM.print("global_sync");
    CHECK_CUDA_ERROR*/

    GraphStd<> graph;
    graph.read(argv[1], parsing_prop::PRINT_INFO | parsing_prop::RM_SINGLETON);

    int  size       = graph.nV();
    auto prefixsum  = graph.csr_out_offsets();
    int  ceil_total = xlib::upper_approx(graph.nE(), ITEMS_PER_BLOCK);
    //--------------------------------------------------------------------------
    //   HOST
    auto h_pos    = new int[ceil_total];
    auto h_offset = new int[ceil_total];
    for (int i = 0, k = 0; i < size; i++) {
        for (int j = prefixsum[i]; j < prefixsum[i + 1]; j++) {
            h_pos[k]      = i;
            h_offset[k++] = j - prefixsum[i];
        }
    }
    for (int i = prefixsum[size]; i < ceil_total; i++)
        h_pos[i] = -1;
    //--------------------------------------------------------------------------
    int num_merge = graph.nE() + graph.nV();

    /*for (int i = 0; i < 70; i++) {
        auto range_start = ::merge_path_search(prefixsum, size + 1,
                                             natural_iterator, prefixsum[size],
                                             min(i * THREAD_ITEMS, num_merge));
        auto   range_end = ::merge_path_search(prefixsum, size + 1,
                                             natural_iterator, prefixsum[size],
                                             min((i + 1) * THREAD_ITEMS, num_merge));
        std::cout << i << "\t" << min(i * THREAD_ITEMS, num_merge)
                  << "\t(" << range_start.x << ", " << range_start.y << ")\t\t("
                  << range_end.x  << "\t" << range_end.y << ")" <<std::endl;
    }*/

    if (PRINT) {
        graph.print_raw();
        std::cout << "Experted results:\n\n";
        xlib::printArray(prefixsum, size + 1);
        xlib::printArray(h_pos, prefixsum[size]);
        xlib::printArray(h_offset, prefixsum[size]);
    }

    int* d_prefixsum, *d_pos, *d_offset, *d_partitions;
    int merge_blocks           = xlib::ceil_div<ITEMS_PER_BLOCK>(num_merge);
    int merge_block_partitions = xlib::ceil_div<BLOCK_SIZE>(merge_blocks);

    int num_blocks           = xlib::ceil_div<ITEMS_PER_BLOCK>(graph.nE());
    int num_block_partitions = xlib::ceil_div<BLOCK_SIZE>(num_blocks);

    std::cout <<   "   THREAD_ITEMS:    " << THREAD_ITEMS
              << "\n   ITEMS_PER_BLOCK: " << ITEMS_PER_BLOCK
              << "\n   Total items:     " << graph.nE()
              << "\n   Num blocks:      " << num_blocks
              << "\n   Num Merges Part.: " << merge_blocks
              << "\n" << std::endl;

    cuMalloc(d_prefixsum, size + 1);
    cuMalloc(d_pos, ceil_total);
    cuMalloc(d_offset, ceil_total);
    cuMalloc(d_partitions, merge_blocks + 1);
    cuMemcpyToDevice(prefixsum, size + 1, d_prefixsum);
    cuMemset0x00(d_pos, ceil_total);
    cuMemset0x00(d_offset, ceil_total);
    cuMemset0x00(d_partitions, num_blocks + 1);
    //--------------------------------------------------------------------------
    TM.start();

    /*binarySearchLBPartition <ITEMS_PER_BLOCK>
        <<< num_block_partitions, BLOCK_SIZE >>>
        (d_prefixsum, size + 1, d_partitions, num_blocks);*/

    mergePathLBPartition <ITEMS_PER_BLOCK>
        <<< merge_block_partitions, BLOCK_SIZE >>>
        (d_prefixsum, size, graph.nE(), num_merge, d_partitions, merge_blocks);

    TM.stop();
    TM.print("Partition:  ");

    //::gpu::printArray(d_partitions + merge_blocks - 5, 6);
    //::gpu::printArray(d_partitions, 5);

    TM.start();

    //MergePathTest<ITEMS_PER_BLOCK, BLOCK_SIZE> <<< num_blocks, BLOCK_SIZE >>>
    //    (d_prefixsum, size + 1, d_pos, d_offset);

    //MergePathTest2<ITEMS_PER_BLOCK, BLOCK_SIZE> <<< num_blocks, BLOCK_SIZE >>>
    //    (d_partitions, num_blocks, d_prefixsum, size + 1, d_pos, d_offset);

    MergePathTest2<ITEMS_PER_BLOCK, BLOCK_SIZE> <<< merge_blocks, BLOCK_SIZE >>>
        (d_partitions, merge_blocks, d_prefixsum, size + 1, d_pos, d_offset);

    TM.stop();
    TM.print("BinarySearch:  ");

    CHECK_CUDA_ERROR
    //--------------------------------------------------------------------------
    if (PRINT) {
        std::cout << "Results:\n\n";
        ::gpu::printArray(d_pos,    graph.nE());
        ::gpu::printArray(d_offset, graph.nE());
    }
    //xlib::printArray(h_pos, 100);
    //::gpu::printArray(d_pos,  100);

    std::cout << "\n Check Positions: "
              << ::gpu::equal(h_pos, h_pos + graph.nE(), d_pos)
              //<< "\n   Check Offsets: "
              //<< ::gpu::equal(h_offset, h_offset + graph.nE(), d_offset)
              << "\n" << std::endl;

    //L1:

#if defined(ENABLE_MGPU)
    using namespace mgpu;
    standard_context_t context;

    int    num_segments = graph.nV();
    int           count = graph.nE();
    const auto&  vector = std::vector<int>(prefixsum, prefixsum + num_segments);
    mem_t<int> segments = to_mem(vector, context);

    mem_t<int> lbs(count, context);
    TM.start();

    load_balance_search(count, segments.data(), num_segments, lbs.data(),
                        context);

    TM.stop();
    TM.print("ModernGPU:  ");

    auto lbs_host = from_mem(lbs);
    std::cout << "\n   Check Offsets: "
              << std::equal(h_pos, h_pos + graph.nE(), lbs_host.data())
              << "\n" << std::endl;
#endif
}
