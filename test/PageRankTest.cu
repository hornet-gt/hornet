/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/PageRank/PageRank.cuh"
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornet_alg;
    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                           graph.out_edges_ptr());
    HornetGPU hornet_graph(hornet_init);

    StaticPageRank page_rank(hornet_graph, 5, 0.001);

    Timer<DEVICE> TM;
    TM.start();

    page_rank.run();

    TM.stop();
    TM.print("PageRank");

    auto is_correct = page_rank.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return !is_correct;
}
