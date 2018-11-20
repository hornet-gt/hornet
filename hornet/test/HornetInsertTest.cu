#include "Hornet.hpp"
#include "Core/GPUHornet/BatchUpdate.cuh"
#include "Util/BatchFunctions.hpp"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>

using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;
using namespace gpu::batch_property;

using HornetGPU = hornets_nest::gpu::Hornet<EMPTY, EMPTY>;

void exec(int argc, char* argv[]);

/**
 * @brief Example tester for Hornet
 */
int main(int argc, char* argv[]) {
    exec(argc, argv);
    cudaDeviceReset();
}

void exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    xlib::device_info();

    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);
    //--------------------------------------------------------------------------
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGPU hornet_gpu(hornet_init);
    std::cout << "------------------------------------------------" <<std::endl;
    using namespace batch_gen_property;

    vid_t* batch_src, *batch_dst;
    int batch_size = std::stoi(argv[2]);

    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);

    generateBatch(graph,
            batch_size, batch_src, batch_dst,
            BatchGenType::INSERT);

    gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);

    hornet_gpu.reserveBatchOpResource(batch_size);

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    Timer<DEVICE> TM(3);
    TM.start();
    hornet_gpu.insertEdgeBatch(batch_update);

    TM.stop();

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    TM.print("Insertion " + std::to_string(batch_size) + ":  ");

    cuFreeHost(batch_src);
    cuFreeHost(batch_dst);
}
