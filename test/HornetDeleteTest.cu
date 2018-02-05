#include "Hornet.hpp"
#include "Core/GPUHornet/BatchUpdate.cuh"
#include "Util/BatchFunctions.hpp"
#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo

using namespace hornets_nest;
using HornetGPU = hornets_nest::gpu::Hornet<EMPTY, EMPTY>;

void exec(int argc, char* argv[]);
void deleteBatchTest(HornetGPU &hornet, graph::GraphStd<vid_t, eoff_t> &graph, int batch_size);

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

    deleteBatchTest(hornet, graph, std::stoi(argv[2]));
}

void deleteBatchTest(HornetGPU &hornet, graph::GraphStd<vid_t, eoff_t> &graph, int batch_size) {
    vid_t* batch_src, *batch_dst;
    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);

    generateBatch(graph,
            batch_size, batch_src, batch_dst,
            BatchGenType::REMOVE,
            batch_gen_property::UNIQUE);

    gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);

    hornet.allocateEdgeDeletion(batch_size,
                                     gpu::batch_property::OUT_OF_PLACE);
    hornet.deleteEdgeBatch(batch_update);

    cuFreeHost(batch_src);
    cuFreeHost(batch_dst);
}
