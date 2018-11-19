#include <XLib.hpp>
#include <iostream>

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void MergePathTest(const int* __restrict__ d_prefixsum,
                   int                     prefixsum_size,
                   int* __restrict__       d_pos,
                   int* __restrict__       d_offset) {
    __shared__ int smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&](int pos, int offset) {
                             int index = d_prefixsum[pos] + offset;
                             d_pos[index]    = pos;
                             d_offset[index] = offset;
                        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_prefixsum, prefixsum_size, smem, lambda);
}

using namespace xlib;
using namespace graph;
using namespace timer;

const bool PRINT = false;
const int BLOCK_SIZE = 32;

int main(int argc, char* argv[]) {
    const int         SMEM = xlib::SMemPerBlock<BLOCK_SIZE, int>::value;
    const int THREAD_ITEMS = xlib::SMemPerThread<int, BLOCK_SIZE>::value;

    Timer<DEVICE> TM;
    GraphStd<> graph;
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
;
    int num_blocks = xlib::ceil_div<SMEM>(graph.nE());

    std::cout <<   "THREAD_ITEMS: " << THREAD_ITEMS
              << "\n  SMEM_ITEMS: " << SMEM
              << "\n total items:"  << graph.nE()
              << "\n  num blocks: " << num_blocks << std::endl;

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
    TM.print("BinarySearch");

    CHECK_CUDA_ERROR
    //--------------------------------------------------------------------------
    if (print) {
        cu::printArray(d_pos, graph.nE());
        cu::printArray(d_offset, graph.nE());
    }
    std::cout <<   "Check Positions: "
              << cu::equal(h_pos, h_pos + graph.nE(), d_pos)
              << "\n  Check Offsets: "
              << cu::equal(h_offset, h_offset + graph.nE(), d_offset)
              << "\n" << std::endl;
}
