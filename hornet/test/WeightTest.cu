#include "Hornet.hpp"
#include <Graph/GraphWeight.hpp>

using weight_t = float;

using namespace hornets_nest;

/**
 * @brief Ensure that the GraphIO GraphWeight class can read weights
 * @author Kasimir Gabert <kasimir@gatech.edu>
 */
int exec(int argc, char* argv[]) {
    graph::GraphWeight<vid_t, eoff_t, weight_t> graph;
    graph.read(argv[1]);
    graph.print();

    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
#if defined(RMM_WRAPPER)
    gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

