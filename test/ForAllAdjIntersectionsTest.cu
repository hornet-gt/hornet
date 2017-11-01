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

using namespace timer;
using namespace hornets_nest;

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

struct GetAttr {
    OPERATOR(Vertex& src, Edge& edge) {
        if (edge.src_id() == 2170) {
            printf("edge %d -> %d : \n", edge.src_id(), edge.dst_id());
        }
    }
};

int main(int argc, char* argv[]) {

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);
    graph.print();

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
    hornet_graph.print();

    Timer<DEVICE> TM(5);
    cudaProfilerStart();
    TM.start();

    // Get the edge values
    //load_balacing::VertexBased1 load_balancing { hornet_graph };
    //forAllEdges(hornet_graph, GetAttr { }, load_balancing);

    TM.stop();
    cudaProfilerStop();
    TM.print("ForAllAdjIntersections Time");

    return 0;
}
