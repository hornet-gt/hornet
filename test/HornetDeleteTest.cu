#include "Hornet.hpp"
#include "Core/GPUHornet/BatchUpdate.cuh"
#include "Util/BatchFunctions.hpp"
#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo
#include <Device/Util/Timer.cuh>             //xlib::Timer
#include <string>
#include <algorithm>                    //std:.generate

using namespace timer;
using namespace hornets_nest;
using HornetGPU = hornets_nest::gpu::Hornet<EMPTY, EMPTY>;

#define RANDOM

void exec(int argc, char* argv[]);
void deleteBatchTest(HornetGPU &hornet,
        graph::GraphStd<vid_t, eoff_t> &graph,
        int batch_size,
        const bool print_debug = false);

int main(int argc, char* argv[]) {
    exec(argc, argv);
    cudaDeviceReset();
    return 0;
}

void exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    xlib::device_info();

    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);

    HornetInit hornet_init(graph.nV(), graph.nE(),
            graph.csr_out_offsets(), graph.csr_out_edges());

    HornetGPU hornet(hornet_init);

    deleteBatchTest(hornet, graph, std::stoi(argv[2]), false);
}

#ifndef RANDOM
    const vid_t karate_src[] = {
0, 0, 0, 0, 1, 1, 9, 16
    };

    const vid_t karate_dst[] = {
3, 5, 8, 19, 17, 21, 2, 6
    };
    const int karate_batch_size = 8;
#endif

void deleteBatchTest(HornetGPU &hornet,
        graph::GraphStd<vid_t, eoff_t> &graph,
        int batch_size,
        const bool print_debug) {
    vid_t* batch_src, *batch_dst;

#ifdef RANDOM
    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);

    generateBatch(graph,
            batch_size, batch_src, batch_dst,
            BatchGenType::REMOVE,
            batch_gen_property::UNIQUE);
#else
    batch_size = karate_batch_size;

    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);

    std::copy(karate_src, karate_src + batch_size, batch_src);
    std::copy(karate_dst, karate_dst + batch_size, batch_dst);
#endif

    gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);
    if (print_debug) {batch_update.print();}

    std::cout<<"=======\n";

    timer::Timer<timer::DEVICE> TM(3);
    TM.start();
    hornet.allocateEdgeDeletion(batch_size,
                                     gpu::batch_property::IN_PLACE);
    TM.stop();
    TM.print("Deletion " + std::to_string(batch_size) + ":  ");
    if (print_debug) {hornet.print();}

    hornet.deleteEdgeBatch(batch_update);

    std::cout<<"=======\n";

    if (print_debug) {hornet.print();}

    cuFreeHost(batch_src);
    cuFreeHost(batch_dst);
}
