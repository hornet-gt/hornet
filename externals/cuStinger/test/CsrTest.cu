#include "Core/GPUCsr/Csr.cuh"
#include "Core/GPUCsr/CsrDevice.cuh"
#include "Core/GPUCsr/CsrTypes.cuh"
#include "GraphIO/GraphStd.hpp"        //GraphStd

using namespace hornet;

using HornetCSR = csr::Hornet<EMPTY, EMPTY>;

/**
 * @brief Example tester for cuSTINGER.
 * Loads an input graph, creates a batches of edges, inserts them into the
 * graph, and then removes them from the graph.
 */
int main(int argc, char* argv[]) {
    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);
    graph.print();

    HornetInit hornet_init(graph.nV(), graph.nE(),
                           graph.out_offsets_ptr(), graph.out_edges_ptr());

    HornetCSR csr_graph(hornet_init);
    csr_graph.print();
}
