#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
//#include <Operator++.cuh>
//#include <Queue/TwoLevelQueue.cuh>
#include <Static/Dummy/Dummy.cuh>
//#include <HornetAlg.hpp>

//using hornets_nest::CommandLineParam;
int main(int argc, char * argv[]) {
    using namespace hornets_nest;
    graph::GraphStd<vert_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
    Dummy bfs_top_down(hornet_graph);
    return 0;
}
