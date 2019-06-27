#include <Hornet.hpp>
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>
#include <Core/Static/Static.cuh>

//using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;

using vert_t = int;
using eoff_t = int;
using HornetGPU = hornet::gpu::Hornet<vert_t>;
using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using Update = hornet::gpu::BatchUpdate<vert_t>;
using Init = hornet::HornetInit<vert_t>;
using hornet::SoAData;
using hornet::TypeList;
using hornet::DeviceType;
using hornet::print;
//using hornet::generateBatchData;

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

    HornetGPU hornet_gpu(hornet_init);
    auto init_coo = hornet_gpu.getCOO(true);

    hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits;
    auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
    Update batch_update(randomBatch);
    hornet_gpu.insert(batch_update);
    auto inst_coo = hornet_gpu.getCOO(true);
    init_coo.append(randomBatch);
    init_coo.sort();

    hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_init_coo = init_coo;
    hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_inst_coo = inst_coo;

    auto *s = host_init_coo.srcPtr();
    auto *d = host_init_coo.dstPtr();
    auto *S = host_inst_coo.srcPtr();
    auto *D = host_inst_coo.dstPtr();
    auto len = host_init_coo.size();
    bool err = false;
    if (host_inst_coo.size() != host_init_coo.size()) {
      err = true;
      std::cerr<<"\nInit Size "<<host_init_coo.size()<<" != Combined size "<<host_inst_coo.size()<<"\n";
      len = std::min(host_init_coo.size(), host_inst_coo.size());
    }
    for (int i = 0; i < len; ++i) {
      if ((s[i] != S[i]) || (d[i] != D[i])) {
        err = true;
        std::cout<<"ERR : ";
        std::cout<<s[i]<<" "<<d[i]<<"\t";
        std::cout<<"\t\t";
        std::cout<<S[i]<<" "<<D[i];
        std::cout<<"\n";
      }
    }
    if (!err) {
      std::cout<<"PASSED\n";
    } else {
      std::cout<<"NOT PASSED\n";
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

