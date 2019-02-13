#include "Hornet.hpp"
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>

//using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;

using vert_t = int;
using HornetGPU = hornet::gpu::Hornet<vert_t>;
using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using Update = hornet::gpu::BatchUpdate<vert_t>;
using Init = hornet::HornetInit<vert_t>;

/**
 * @brief Example tester for Hornet
 */
int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    xlib::device_info();

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(argv[1]);
    //--------------------------------------------------------------------------
    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGPU hornet_gpu(hornet_init);
    std::cout << "------------------------------------------------" <<std::endl;
    using namespace hornets_nest::batch_gen_property;
    using namespace hornets_nest::host;

    vert_t* batch_src, *batch_dst;
    int batch_size = std::stoi(argv[2]);

    allocatePageLocked(batch_src, batch_size);
    allocatePageLocked(batch_dst, batch_size);

    generateBatch(graph,
            batch_size, batch_src, batch_dst,
            hornets_nest::BatchGenType::INSERT);
    UpdatePtr ptr(batch_size, batch_src, batch_dst);

    Update batch_update(ptr);

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    Timer<DEVICE> TM(3);
    TM.start();
    hornet_gpu.insert(batch_update);

    TM.stop();

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    TM.print("Insertion " + std::to_string(batch_size) + ":  ");

    freePageLocked(batch_dst, batch_src);

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

