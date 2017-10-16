/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);
    //graph.print();

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                           graph.out_edges_ptr());

    HornetGraph hornet_graph(hornet_init);
    //hornet_graph.print();

    BfsTopDown2 bfs_top_down(hornet_graph);

    bfs_top_down.set_parameters(graph.max_out_degree_id());

    Timer<DEVICE> TM;
    cudaProfilerStart();
    TM.start();

    bfs_top_down.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("TopDown2");

    auto is_correct = bfs_top_down.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return !is_correct;
}
