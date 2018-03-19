#include "Hornet.hpp"
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

#define RANDOM
#ifndef RANDOM
    const vid_t karate_src[] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    };

    const vid_t karate_dst[] = {
100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200
    };
    const int karate_batch_size = 100;
#endif

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
    //--------------------------------------------------------------------------
    using namespace batch_gen_property;

    vid_t* batch_src, *batch_dst;
    int batch_size = std::stoi(argv[2]);

#ifdef RANDOM
    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);

    generateBatch(graph,
            batch_size, batch_src, batch_dst,
            BatchGenType::INSERT);
            //batch_gen_property::UNIQUE);
#else
    batch_size = karate_batch_size;

    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);

    std::copy(karate_src, karate_src + batch_size, batch_src);
    std::copy(karate_dst, karate_dst + batch_size, batch_dst);
#endif

    gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);

    //batch_update.print();

    hornet_gpu.reserveBatchOpResource(batch_size,
                                     IN_PLACE | REMOVE_CROSS_DUPLICATE | REMOVE_BATCH_DUPLICATE);

    //hornet_gpu.print();

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    Timer<DEVICE> TM(3);
    TM.start();
    hornet_gpu.insertEdgeBatch(batch_update);

    TM.stop();

    //batch_update.print();
    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    TM.print("Insertion " + std::to_string(batch_size) + ":  ");
    //hornet_gpu.print();

    cuFreeHost(batch_src);
    cuFreeHost(batch_dst);
}
