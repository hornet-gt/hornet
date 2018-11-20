/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/PageRank/PageRank.cuh"
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>

using namespace hornets_nest;

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;


    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1], PRINT_INFO | SORT);
    // CommandLineParam cmd(graph, argc, argv);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());
    HornetGPU hornet_graph(hornet_init);

    StaticPageRank page_rank(hornet_graph, 50, 0.001, 0.85, false);

    Timer<DEVICE> TM;
    TM.start();

    page_rank.run();

    TM.stop();
    TM.print("PR---InputAsIS");


	graph::ParsingProp flag = PRINT_INFO | SORT;
	        graph::GraphStd<vid_t, eoff_t> graphUnDir(UNDIRECTED);
    graphUnDir.read(argv[1],flag);

    HornetInit hornet_init_undir(graphUnDir.nV(), graphUnDir.nE(), graphUnDir.csr_out_offsets(),
                           graphUnDir.csr_out_edges());
    HornetGPU hornet_graph_undir(hornet_init_undir);

    StaticPageRank page_rank_undir(hornet_graph_undir, 50, 0.001,0.85,true);

    TM.start();

    page_rank_undir.run();

    TM.stop();
    TM.print("PR---Undirected---PULL");        	



    return 0;
}
