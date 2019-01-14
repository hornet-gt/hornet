/**
 * @brief Brim test program
 * @file
 */
//#include "Static/ShortestPath/SSSP.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphWeight.hpp>
#include <Graph/Brim.hpp>
#include <BasicTypes.hpp>
#include <Device/Util/Timer.cuh>

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace graph;
    using namespace hornets_nest;

    graph::GraphWeight<vid_t, eoff_t, int> graph;
    graph.read(argv[1]);

    Brim<vid_t, eoff_t, int> brim(graph);

    Timer<DEVICE> TM;
    TM.start();

    brim.run();

    TM.stop();
    TM.print("Brim");

    std::cout << "MPG Check: " << brim.check() << std::endl;
	brim.check_from_file(argv[2]);

    /*auto is_correct = brim.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return is_correct;*/

    return 0;
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

