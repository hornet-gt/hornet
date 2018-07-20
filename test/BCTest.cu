/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BetweennessCentrality/bc.cuh"
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

using namespace std;
using namespace graph;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    // GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    CommandLineParam cmd(graph, argc, argv,false);

    // graph.read(argv[1], SORT | PRINT_INFO);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);

    BCCentrality bc(hornet_graph);

	vid_t root = graph.max_out_degree_id();
	if (argc==3)
	  root = atoi(argv[2]);
    // root = 226410;
    cout << "Root is " << root << endl;
    bc.reset();

    bc.setRoot(root);


    Timer<DEVICE> TM;
    cudaProfilerStart();
    TM.start();

    bc.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("BCCentrality");

    // auto is_correct = bc.validate();
    // std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    // return !is_correct;
}
