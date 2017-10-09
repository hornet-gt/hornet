#include "Hornet.hpp"
#include "Util/BatchFunctions.hpp"
#include "Host/FileUtil.hpp"            //xlib::extract_filepath_noextension
#include "Device/CudaUtil.cuh"          //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include "Core/GPU/BatchUpdate.cuh"
#include <cuda_profiler_api.h>
//nvprof --profile-from-start off --log-file log.txt --print-gpu-trace

using namespace hornet;
using namespace timer;
using namespace std::string_literals;

using HornetGPU = hornet::gpu::Hornet<EMPTY, EMPTY>;

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

    graph::GraphStd<vid_t, eoff_t> graph(REVERSE);
    graph.read(argv[1]);
    //graph.print();
    //if (param.binary)
    //    graph.writeBinary(xlib::extract_filepath_noextension(argv[1]) + ".bin");
    // graph.writeDimacs10th(xlib::extract_filepath_noextension(argv[1]) + ".graph");
    //graph.writeMarket(xlib::extract_filepath_noextension(argv[1]) + ".mtx");
    //--------------------------------------------------------------------------
    auto weights = new int[graph.nE()];
    std::iota(weights, weights + graph.nE(), 0);
    //--------------------------------------------------------------------------
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                           graph.out_edges_ptr());
    //hornet_init.insertEdgeData(weights);

    HornetGPU hornet_gpu(hornet_init);
    //hornet_gpu.mem_manager_info();
    hornet_gpu.print();
    //return;
    //hornet_gpu.check_sorted_adjs();
    std::cout << "------------------------------------------------" <<std::endl;
    //--------------------------------------------------------------------------
    using namespace batch_gen_property;

    if (argc == 3) {
        int batch_size = std::stoi(argv[2]);
        vid_t* batch_src, *batch_dst;
        cuMallocHost(batch_src, batch_size);
        cuMallocHost(batch_dst, batch_size);

        generateBatch(graph, batch_size, batch_src, batch_dst,
                      BatchGenType::INSERT, UNIQUE); //| PRINT
        //vid_t batch_src[] = { 0, 0, 2 };
        //vid_t batch_dst[] = { 2, 6, 7 };
        gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);

        batch_update.print();
        std::cout << "------------------------------------------------" <<std::endl;

        using namespace hornet::gpu::batch_property;
        //hornet_gpu.allocateEdgeInsertion(batch_size,
        //                                 IN_PLACE | REMOVE_CROSS_DUPLICATE);
        hornet_gpu.allocateEdgeDeletion(batch_size,
                                         IN_PLACE | REMOVE_CROSS_DUPLICATE);

        cudaProfilerStart();
        Timer<DEVICE> TM(3);
        TM.start();

        //hornet_gpu.insertEdgeBatch(batch_update);
        //hornet_gpu.deleteEdgeBatch(batch_update);

        TM.stop();
        TM.print("Insertion "s + std::to_string(batch_size) + ":  ");
        cudaProfilerStop();
        //hornet_gpu.check_sorted_adjs();
        //delete[] batch_src;
        //delete[] batch_dst;
        cuFreeHost(batch_src);
        cuFreeHost(batch_dst);
        hornet_gpu.print();
    }
    delete[] weights;
}
