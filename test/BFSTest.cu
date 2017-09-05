/**
 * @brief Breadth-first Search Top-Down test program (C++11 Style APIs)
 * @file
 */
#include "Static/BreadthFirstSearch/TopDown.cuh"
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
//#include <cuda_profiler_api.h> //--profile-from-start off

using namespace hornet_alg;

int main(int argc, char* argv[]) {
    using namespace timer;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                          graph.out_edges_ptr());

    //HornetCSR hornet_graph(hornet_init);
    HornetGPU hornet_graph(hornet_init);

    BfsTopDown2 bfs_top_down(hornet_graph);

    //bfs_top_down.set_parameters(graph.max_out_degree_id());
    bfs_top_down.set_parameters(0);

    Timer<DEVICE> TM;
    TM.start();
    //cuProfilerStart();

    bfs_top_down.run();

    //cuProfilerStop();
    TM.stop();
    TM.print("TopDown");

    auto is_correct = bfs_top_down.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return is_correct;
}
