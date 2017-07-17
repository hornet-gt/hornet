/**
 * @brief Breadth-first Search Top-Down test program (C++11 Style APIs)
 * @file
 */
#include "Static/BreadthFirstSearch/TopDown++.cuh"

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace custinger;
    using namespace custinger_alg;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);

    cuStingerInit custinger_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                                 graph.out_edges_ptr());

    cuStinger custiger_graph(custinger_init);

    BfsTopDown2 bfs_top_down(custiger_graph);
    bfs_top_down.set_parameters(graph.max_out_degree_id());
    Timer<DEVICE> TM;
    TM.start();

    bfs_top_down.run();

    TM.stop();
    TM.print("TopDown");

    auto is_correct = bfs_top_down.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return is_correct;
}
