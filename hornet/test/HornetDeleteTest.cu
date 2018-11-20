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
void deleteBatch(HornetGPU &hornet,
        const vid_t * src,
        const vid_t * dst,
        const int batch_size,
        const bool print_debug = false);
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

void deleteBatch(HornetGPU &hornet,
        vid_t * src,
        vid_t * dst,
        const int batch_size,
        const bool print_debug) {
    Timer<DEVICE> TM(3);
    gpu::BatchUpdate batch_update(src, dst, batch_size);


    if (print_debug) {
        batch_update.print();
        std::cout<<"ne: "<<hornet.nE()<<"\n=======\n";
        hornet.print();
    }

    TM.start();
    hornet.deleteEdgeBatch(batch_update);
    TM.stop();

    if (print_debug) {
        std::cout<<"ne: "<<hornet.nE()<<"\n=======\n";
        hornet.print();
    }

    TM.print("Deletion " + std::to_string(batch_size) + ":  ");
}

void deleteBatchTest(HornetGPU &hornet,
        graph::GraphStd<vid_t, eoff_t> &graph,
        int batch_size,
        const bool print_debug) {
    #ifndef RANDOM
    vid_t batch_src[] = {0, 2, 23, 32, 32, 33, 33, 33};
    vid_t batch_dst[] = {31, 27, 27, 23, 31, 23, 27, 31};
    batch_size = 8;

    #else
    vid_t* batch_src, *batch_dst;
    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);
    generateBatch(graph,
            batch_size, batch_src, batch_dst,
            BatchGenType::INSERT);
    #endif

    hornet.reserveBatchOpResource(batch_size,
                                     gpu::batch_property::IN_PLACE | gpu::batch_property::REMOVE_BATCH_DUPLICATE);
    deleteBatch(hornet, batch_src, batch_dst, batch_size, print_debug);

    #ifndef RANDOM
    #else
    cuFreeHost(batch_src);
    cuFreeHost(batch_dst);
    #endif
}
