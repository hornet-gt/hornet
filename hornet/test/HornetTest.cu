#include "Hornet.hpp"
#include "StandardAPI.hpp"
#include "Core/GPUHornet/BatchUpdate.cuh"
#include "Util/BatchFunctions.hpp"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>
//nvprof --profile-from-start off --log-file log.txt --print-gpu-trace

using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;

using HornetGPU = hornets_nest::gpu::Hornet<EMPTY, EMPTY>;

/**
 * @brief Example tester for Hornet
 */
int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    xlib::device_info();

    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);
    auto weights = new int[graph.nE()];
    std::iota(weights, weights + graph.nE(), 0);
    //--------------------------------------------------------------------------
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGPU hornet_gpu(hornet_init);
    std::cout << "------------------------------------------------" <<std::endl;
    //--------------------------------------------------------------------------
    using namespace batch_gen_property;

    if (argc == 3) {
        int batch_size = std::stoi(argv[2]);

#ifdef TEST
        batch_size = 100;
#endif
        vid_t* batch_src, *batch_dst;
        host::allocatePageLocked(batch_src, batch_size);
        host::allocatePageLocked(batch_dst, batch_size);
#ifdef TEST
        for (int i = 0; i < batch_size - 10; ++i) {
            batch_src[i] = 33;
            batch_dst[i] = 8;
        }
        for (int i = batch_size - 10; i < batch_size; ++i) {
            batch_src[i] = 33;
            batch_dst[i] = 8;
        }
#else
        generateBatch(graph, batch_size, batch_src, batch_dst,
                      BatchGenType::INSERT, UNIQUE);
#endif
        gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);

        //batch_update.print();
        std::cout << "------------------------------------------------" <<std::endl;

        using namespace gpu::batch_property;

        hornet_gpu.reserveBatchOpResource(batch_size);

        hornet_gpu.print();
        std::cout << "------------------------------------------------" <<std::endl;
        cudaProfilerStart();
        Timer<DEVICE> TM(3);
        TM.start();

        hornet_gpu.insertEdgeBatch(batch_update);
        //hornet_gpu.deleteEdgeBatch(batch_update);

        TM.stop();
        //TM.print("Insertion "s + std::to_string(batch_size) + ":  ");
        cudaProfilerStop();
        //hornet_gpu.check_sorted_adjs();
        //delete[] batch_src;
        //delete[] batch_dst;
        host::freePageLocked(batch_src, batch_dst);
        //batch_update.print();
        hornet_gpu.print();
    }
    delete[] weights;

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

