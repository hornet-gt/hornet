#include "StaticBreadthFirstSearch/TopDown++.cuh"

int main(int argc, char* argv[]) {
    using namespace custinger;
    using namespace custinger_alg;
    cudaSetDevice(1);

    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);

    cuStingerInit custinger_init(graph.nV(), graph.nE(), graph.out_offsets(),
                                 graph.out_edges());

    cuStinger custiger_graph(custinger_init);

    BfsTopDown2 bfs_top_down(custiger_graph);
    bfs_top_down.set_parameters(0);
    bfs_top_down.run();

    auto is_correct = bfs_top_down.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return is_correct;
}
