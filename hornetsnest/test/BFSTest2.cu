/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);


    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    Timer<DEVICE> TM;
    cudaProfilerStart();
    TM.start();
    HornetGraph hornet_graph(hornet_init);
    TM.stop();
    cudaProfilerStop();
    TM.print("Initilization Time:");

    BfsTopDown2 bfs_top_down(hornet_graph);

    vid_t root = graph.max_out_degree_id();
    if (argc==3)
        root = atoi(argv[2]);

    std::cout << "My root is " << root << std::endl;

    bfs_top_down.set_parameters(root);

    cudaProfilerStart();
    TM.start();

    bfs_top_down.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("TopDown2");

    std::cout << "Number of levels is : " << bfs_top_down.getLevels() << std::endl;

    auto is_correct = bfs_top_down.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return !is_correct;
}

int main(int argc, char* argv[]) {
    int ret = 0;
#if defined(RMM_WRAPPER)
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

