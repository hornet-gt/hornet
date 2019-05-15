#include "Hornet.hpp"
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo
#include <Device/Util/Timer.cuh>             //xlib::Timer
#include <string>
#include <algorithm>                    //std:.generate
using namespace timer;
using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;

using vert_t = int;
using HornetGPU = hornet::gpu::Hornet<vert_t>;
//using UpdatePtr = hornet::BatchUpdatePtr<vert_t>;
using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using Update = hornet::gpu::BatchUpdate<vert_t>;
using Init = hornet::HornetInit<vert_t>;

//#define RANDOM

void deleteBatch(HornetGPU &hornet,
        vert_t * src,
        vert_t * dst,
        const int batch_size,
        const bool print_debug) {
    UpdatePtr ptr(batch_size, src, dst);

    Update batch_update(ptr);

    Timer<DEVICE> TM(3);


    if (print_debug) {
        batch_update.print();
        std::cout<<"ne: "<<hornet.nE()<<"\n=======\n";
        hornet.print();
    }

    TM.start();
    hornet.erase(batch_update);
    TM.stop();

    if (print_debug) {
        std::cout<<"ne: "<<hornet.nE()<<"\n=======\n";
        hornet.print();
    }

    TM.print("Deletion " + std::to_string(batch_size) + ":  ");
}

void deleteBatchTest(HornetGPU &hornet,
        graph::GraphStd<vert_t, eoff_t> &graph,
        int batch_size,
        const bool print_debug) {
    #ifndef RANDOM
    vert_t batch_src[] = {1, 5, 2, 4};
    vert_t batch_dst[] = {2, 4, 1, 5};
    batch_size = 4;

    #else
    vert_t* batch_src, *batch_dst;
    host::allocatePageLocked(batch_src, batch_size);
    host::allocatePageLocked(batch_dst, batch_size);
    generateBatch(graph,
            batch_size, batch_src, batch_dst,
            BatchGenType::INSERT);
    #endif

    deleteBatch(hornet, batch_src, batch_dst, batch_size, print_debug);

    #ifndef RANDOM
    #else
    host::freePageLocked(batch_src, batch_dst);
    #endif
}

int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    xlib::device_info();
    graph::GraphStd<vert_t, vert_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT_INFO);
    Init hornet_init(graph.nV(), graph.nE(),
            graph.csr_out_offsets(), graph.csr_out_edges());
    HornetGPU hornet(hornet_init);
    deleteBatchTest(hornet, graph, 4, false);
    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
#if defined(RMM_WRAPPER)
    gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

