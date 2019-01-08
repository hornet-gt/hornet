#include "Hornet.hpp"
#include <Graph/GraphWeight.hpp>

using weight_t = float;

using namespace hornets_nest;

/**
 * @brief Ensure that the GraphIO GraphWeight class can read weights
 * @author Kasimir Gabert <kasimir@gatech.edu>
 */
int main(int argc, char* argv[]) {
#if defined(RMM_WRAPPER)
    size_t init_pool_size = 128 * 1024 * 1024;//128MB
    gpu::initializeRMMPoolAllocation(init_pool_size);
#endif

    graph::GraphWeight<vid_t, eoff_t, weight_t> graph;
    graph.read(argv[1]);
    graph.print();

#if defined(RMM_WRAPPER)
    gpu::finalizeRMMPoolAllocation();
#endif

    return 0;
}
