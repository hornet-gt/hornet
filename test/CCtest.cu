/**
 * @brief Breadth-first Search Top-Down test program (C++11 Style APIs)
 * @file
 */
#include "Static/ConnectedComponents/CC++.cuh"

int main(int argc, char* argv[]) {
    using namespace graph;
    using namespace timer;
    using namespace custinger;
    using namespace custinger_alg;

    GraphStd<vid_t, eoff_t> graph(Structure::UNDIRECTED);
    CommandLineParam(graph, argc, argv);

    cuStingerInit custinger_init(graph.nV(), graph.nE(), graph.out_offsets(),
                                 graph.out_edges());

    graph.print();

    cuStinger custiger_graph(custinger_init);

    CC cc_multistep(custiger_graph);
    Timer<DEVICE> TM;
    TM.start();

    cc_multistep.run();

    TM.stop();
    TM.print("CC");

    auto is_correct = cc_multistep.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return is_correct;
}
