#include <Hornet.hpp>
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
//#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo
//#include <Device/Util/Timer.cuh>             //xlib::Timer
#include <string>
#include <algorithm>                    //std:.generate
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
using hornet::generateBatchData;

//#define RANDOM

void deleteBatch(HornetGPU &hornet,
        vert_t * src,
        vert_t * dst,
        const int batch_size,
        const bool print_debug) {
    UpdatePtr ptr(batch_size, src, dst);

    Update batch_update(ptr);



    if (print_debug) {
        batch_update.print();
        std::cout<<"ne: "<<hornet.nE()<<"\n=======\n";
        hornet.print();
    }

    hornet.erase(batch_update);

    if (print_debug) {
        std::cout<<"ne: "<<hornet.nE()<<"\n=======\n";
        hornet.print();
    }

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

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(argv[1]);
    int batch_size = std::stoi(argv[2]);
    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

    HornetGPU hornet_gpu(hornet_init);
    auto init_coo = hornet_gpu.getCOO(true);

    auto randomBatch = generateBatchData<vert_t>(graph, batch_size, false);
    Update batch_update(randomBatch);
    hornet_gpu.erase(batch_update);

    auto post_erase_coo = hornet_gpu.getCOO(true);

    post_erase_coo.append(randomBatch);
    post_erase_coo.sort();

    SoAData<TypeList<vert_t, vert_t>, DeviceType::HOST> host_init_coo(init_coo.get_num_items());
    host_init_coo.copy(init_coo);
    SoAData<TypeList<vert_t, vert_t>, DeviceType::HOST> host_inst_coo(post_erase_coo.get_num_items());
    host_inst_coo.copy(post_erase_coo);

    auto *s = host_init_coo.get_soa_ptr().get<0>();
    auto *d = host_init_coo.get_soa_ptr().get<1>();
    auto *S = host_init_coo.get_soa_ptr().get<0>();
    auto *D = host_init_coo.get_soa_ptr().get<1>();
    for (int i = 0; i < host_init_coo.get_num_items(); ++i) {
      if ((s[i] != S[i]) || (d[i] != D[i])) {
        std::cout<<"ERR : "<<s[i]<<" "<<d[i]<<"\n";
      }
    }

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";

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

