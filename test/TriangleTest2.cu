/**
 * @brief
 * @file
 */

#include "HornetAlg.hpp"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

#include "Static/TriangleCounting/triangle2.cuh"

using namespace timer;
using namespace hornets_nest;

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;




int main(int argc, char* argv[]) {

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT_INFO);
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
    // hornet_graph.print();
    TriangleCounting tc(hornet_graph);
    tc.init();

    Timer<DEVICE> TM(5);
    cudaProfilerStart();
    TM.start();

    tc.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("ForAllAdjUnions Time");

    return 0;
}
