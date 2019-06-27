#include "Hornet.hpp"
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
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
using eoff_t = int;
using wgt0_t = int;
using wgt1_t = float;
using Init = hornet::HornetInit<vert_t, hornet::EMPTY, hornet::TypeList<wgt0_t, wgt1_t>>;
using HornetGPU = hornet::gpu::Hornet<vert_t, hornet::EMPTY, hornet::TypeList<wgt0_t, wgt1_t>>;
using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::TypeList<wgt0_t, wgt1_t>, hornet::DeviceType::HOST>;
using Update = hornet::gpu::BatchUpdate<vert_t, hornet::TypeList<wgt0_t, wgt1_t>>;
using hornet::TypeList;
using hornet::DeviceType;

/**
 * @brief Example tester for Hornet
 */
int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(argv[1]);
    int batch_size = std::stoi(argv[2]);
    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

    //Use meta with hornet_init
    std::vector<wgt0_t> edge_meta_0(graph.nE(), 0);
    std::vector<wgt1_t> edge_meta_1(graph.nE(), 1);
    hornet_init.insertEdgeData(edge_meta_0.data(), edge_meta_1.data());
    HornetGPU hornet_gpu(hornet_init);
    auto init_coo = hornet_gpu.getCOO(true);

    hornet::RandomGenTraits<TypeList<wgt0_t, wgt1_t>> cooGenTraits;
    auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
    Update batch_update(randomBatch);

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    Timer<DEVICE> TM(3);
    TM.start();
    hornet_gpu.insert(batch_update);

    TM.stop();

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    TM.print("Insertion " + std::to_string(batch_size) + ":  ");

    auto inst_coo = hornet_gpu.getCOO(true);
    init_coo.append(randomBatch);
    init_coo.sort();

    std::cout<<"Creating multimap for testing correctness...";
    auto init_coo_map = getHostMMap(init_coo);
    auto inst_coo_map = getHostMMap(inst_coo);
    std::cout<<"...Done!\n";
    if (inst_coo_map == inst_coo_map) {
      std::cout<<"Passed\n";
    } else {
      std::cout<<"Failed\n";
    }

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

