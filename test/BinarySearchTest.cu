#include "Host/Numeric.hpp"
#include "Device/Util/DeviceProperties.cuh"
#include "Device/Util/PrintExt.cuh"
#include "Device/Util/Algorithm.cuh"
#include "Device/Primitives/BinarySearchLB.cuh"
#include "Device/Primitives/impl/BinarySearchLB2.i.cuh"
#include "Device/Primitives/impl/MergePathLB.i.cuh"
#include "Device/Util/Timer.cuh"
#include <Graph/GraphBase.hpp>
#include <Graph/GraphStd.hpp>
#include <iostream>

#define ENABLE_MGPU

#if defined(ENABLE_MGPU)
    #include <moderngpu/kernel_load_balance.hxx>
#endif

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void MergePathTest(const int* __restrict__ d_prefixsum,
                   int                     prefixsum_size,
                   int* __restrict__       d_pos,
                   int* __restrict__       d_offset) {
    __shared__ int smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&](int pos, int offset) {
                             int index      = d_prefixsum[pos] + offset;
                             d_pos[index]   = pos;
                            // d_offset[index] = offset;
                        };
    xlib::binarySearchLB<BLOCK_SIZE, ITEMS_PER_BLOCK / BLOCK_SIZE>
    //xlib::binarySearchLBAllPosNoDup<BLOCK_SIZE, ITEMS_PER_BLOCK / BLOCK_SIZE>
        (d_prefixsum, prefixsum_size, smem, lambda);
}

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void MergePathTest2(const int* __restrict__ d_partitions,
                    int                     num_partitions,
                    const int* __restrict__ d_prefixsum,
                    int                     prefixsum_size,
                    int* __restrict__       d_pos,
                    int* __restrict__       d_offset) {
    __shared__ int smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&](int pos, int offset, int index2) {
                            // int index = d_prefixsum[pos] + offset;
                             //printf("t: %d\t%d\t%d\n", threadIdx.x, index, index2);
                             //assert(index == index2);
                             d_pos[index2]    = pos;
                             //d_offset[index] = offset;
                        };
    xlib::binarySearchLB2<BLOCK_SIZE, ITEMS_PER_BLOCK / BLOCK_SIZE, true>
        (d_partitions, num_partitions, d_prefixsum, prefixsum_size, smem, lambda);
}

template<typename itA_t, typename itB_t>
int2 merge_path_search(const itA_t& A, int a_len,
                       const itB_t& B, int b_len,
                       int diagonal) {
    int x_min = max(diagonal - b_len, 0);
    int x_max = min(diagonal, a_len);

    while (x_min < x_max) {
        int pivot = (x_max + x_min) / 2u;
        if (A[pivot] <= B[diagonal - pivot - 1])
            x_min = pivot + 1;
        else
            x_max = pivot;
    }
    return make_int2(min(x_min, a_len), diagonal - x_min);
}



class NaturalIterator {
public:
    int operator[](int index) const {
        return index;
    }
};

using namespace xlib;
using namespace graph;
using namespace timer;

const bool PRINT      = false;
const int  BLOCK_SIZE = 128;

int main(int argc, char* argv[]) {
    /*NaturalIterator natural_iterator;
    const int NUM_THREADS = 5;
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

    const int THREAD_ITEMS    = 11;
    const int ITEMS_PER_BLOCK = BLOCK_SIZE * THREAD_ITEMS;

    Timer<DEVICE, micro> TM;
    GraphStd<> graph(structure_prop::UNDIRECTED);
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
    if (PRINT) {
        graph.print_raw();
        std::cout << "Experted results:\n\n";
        xlib::printArray(prefixsum, size + 1);
        xlib::printArray(h_pos, prefixsum[size]);
        xlib::printArray(h_offset, prefixsum[size]);
    }

    int* d_prefixsum, *d_pos, *d_offset, *d_partitions;
    int num_merge            = graph.nE() + graph.nV();
    int merge_blocks         = xlib::ceil_div<ITEMS_PER_BLOCK>(num_merge);
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

    //blockPartition<ITEMS_PER_BLOCK> <<< num_block_partitions, BLOCK_SIZE >>>
    //    (d_prefixsum, size + 1, d_partitions, num_blocks);

    blockPartition2<ITEMS_PER_BLOCK> <<< merge_block_partitions, BLOCK_SIZE >>>
        (d_prefixsum, size, graph.nE(), num_merge, d_partitions, merge_blocks);

    TM.stop();
    TM.print("Partition:  ");

    ::gpu::printArray(d_partitions, 100);
    /*auto h_partition = new int[merge_blocks + 1];
    cuMemcpyToHost(d_partitions, merge_blocks + 1, h_partition);
    for (int i = 0; i < 10; i++)
        std::cout << prefixsum[h_partition[i]] << " ";
    std::cout << "\n\n";*/
    goto L1;

    TM.start();

    //MergePathTest<ITEMS_PER_BLOCK, BLOCK_SIZE> <<< num_blocks, BLOCK_SIZE >>>
    //    (d_prefixsum, size + 1, d_pos, d_offset);

    MergePathTest2<ITEMS_PER_BLOCK, BLOCK_SIZE> <<< num_blocks, BLOCK_SIZE >>>
        (d_partitions, num_blocks, d_prefixsum, size + 1, d_pos, d_offset);

    TM.stop();
    TM.print("BinarySearch:  ");

    CHECK_CUDA_ERROR
    //--------------------------------------------------------------------------
    if (PRINT) {
        std::cout << "Results:\n\n";
        ::gpu::printArray(d_pos,    graph.nE());
        ::gpu::printArray(d_offset, graph.nE());
    }
    std::cout << "\n Check Positions: "
              << ::gpu::equal(h_pos, h_pos + graph.nE(), d_pos)
              //<< "\n   Check Offsets: "
              //<< ::gpu::equal(h_offset, h_offset + graph.nE(), d_offset)
              << "\n" << std::endl;

    L1:

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
