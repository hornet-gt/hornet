/**
 * @brief Connected-Component test program
 * @file
 */
#include "Static/ConnectedComponents/CC.cuh"
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                           graph.out_edges_ptr());
    HornetGraph hornet_graph(hornet_init);

    CC cc_multistep(hornet_graph);

    Timer<DEVICE> TM;
    TM.start();

    cc_multistep.run();

    TM.stop();
    TM.print("CC");

    auto is_correct = cc_multistep.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return is_correct;
}
