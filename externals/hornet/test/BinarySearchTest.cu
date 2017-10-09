#include "Host/Numeric.hpp"
#include "Device/Definition.cuh"
#include "Device/PrintExt.cuh"
#include "Device/Algorithm.cuh"
#include "Device/BinarySearchLB.cuh"
#include "Device/Timer.cuh"
#include "GraphIO/GraphStd.hpp"
#include <iostream>
//#include <moderngpu/kernel_load_balance.hxx>

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void MergePathTest(const int* __restrict__ d_prefixsum,
                   int                     prefixsum_size,
                   int* __restrict__       d_pos,
                   int* __restrict__       d_offset) {
    __shared__ int smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&](int pos, int offset) {
                             int index = d_prefixsum[pos] + offset;
                             //d_pos[index]    = pos;
                             d_offset[index] = offset;
                        };
    xlib::binarySearchLBAllPosNoDup<BLOCK_SIZE, ITEMS_PER_BLOCK / BLOCK_SIZE>
        (d_prefixsum, prefixsum_size, smem, lambda);
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

const bool     PRINT = false;
const int BLOCK_SIZE = 128;

int main(int argc, char* argv[]) {
    NaturalIterator natural_iterator;
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
    return 0;

    //const int         SMEM = xlib::SMemPerBlock<BLOCK_SIZE, int>::value;
    //const int THREAD_ITEMS = xlib::SMemPerThread<int, BLOCK_SIZE>::value;
    const int THREAD_ITEMS = 12;
    const int         SMEM = BLOCK_SIZE * THREAD_ITEMS;

    Timer<DEVICE> TM;
    GraphStd<> graph(structure_prop::REVERSE);
    graph.read(argv[1]);

    int       size = graph.nV();
    auto prefixsum = graph.out_offsets_ptr();
    int ceil_total = xlib::upper_approx(graph.nE(), SMEM);
    //--------------------------------------------------------------------------
    //   HOST
    auto    h_pos = new int[ceil_total];
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
        xlib::printArray(prefixsum, size + 1);
        xlib::printArray(h_pos, prefixsum[size]);
        xlib::printArray(h_offset, prefixsum[size]);
    }

    int* d_prefixsum, *d_pos, *d_offset;
    int num_blocks = xlib::ceil_div<SMEM>(graph.nE());

    std::cout <<   "   THREAD_ITEMS: " << THREAD_ITEMS
              << "\n     SMEM_ITEMS: " << SMEM
              << "\n    total items: " << graph.nE()
              << "\n     num blocks: " << num_blocks << "\n" << std::endl;

    cuMalloc(d_prefixsum, size + 1);
    cuMalloc(d_pos, ceil_total);
    cuMalloc(d_offset, ceil_total);
    cuMemcpyToDevice(prefixsum, size + 1, d_prefixsum);
    cuMemset0x00(d_pos, ceil_total);
    cuMemset0x00(d_offset, ceil_total);
    //--------------------------------------------------------------------------
    TM.start();

    MergePathTest<SMEM, BLOCK_SIZE> <<< num_blocks, BLOCK_SIZE >>>
        (d_prefixsum, size + 1, d_pos, d_offset);

    TM.stop();
    TM.print("BinarySearch:  ");

    CHECK_CUDA_ERROR
    //--------------------------------------------------------------------------
    if (PRINT) {
        cu::printArray(d_pos, graph.nE());
        cu::printArray(d_offset, graph.nE());
    }
    std::cout << "\n Check Positions: "
              << cu::equal(h_pos, h_pos + graph.nE(), d_pos)
              << "\n   Check Offsets: "
              << cu::equal(h_offset, h_offset + graph.nE(), d_offset)
              << "\n" << std::endl;

    /*using namespace mgpu;
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
    TM.print("ModernGPU:  ");*/
}
