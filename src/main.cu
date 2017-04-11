///@files

#include "Util/BatchFunctions.hpp"
#include "Core/update.cuh"
#include "Core/memoryManager.hpp"
#include "Core/cuStinger.hpp"
#include "Core/BitTree.hpp"
#include "Core/MemoryManagement.hpp"
#include "GraphIO/GraphStd.hpp"
#include "Util/Parameters.hpp"
#include "Util/utils.hpp"
#include "Support/FileUtil.hpp"
#include "Support/Timer2.cuh"


using namespace cu_stinger;
using namespace timer2;

void printcuStingerUtility(cuStinger custing, bool all_info);

/**
 * @brief Example tester for cuSTINGER.
 * Loads an input graph, creates a batches of edges, inserts them into the
 * graph, and then removes them from the graph.
 */
int main(int argc, char* argv[]) {
    xlib::deviceInfo();
    Param param(argc, argv);

    graph::GraphStd<> graph;
    graph.read(argv[1]);

    if (param.binary)
        graph.toBinary(xlib::extract_filepath_noextension(argv[1]) + ".bin");

    cuStingerInitConfig cu_init;

    cu_init.initState  = eInitStateCSR;
    cu_init.maxNV      = graph.nV() + 1;
    cu_init.useVWeight = false;
    cu_init.isSemantic = false;  // Use edge types and vertex types
    cu_init.useEWeight = false;

    cu_init.csrNV  = graph.nV();
    cu_init.csrNE  = graph.nE();
    cu_init.csrOff = graph.out_offsets_array();
    cu_init.csrAdj = graph.out_edges_array();
    cu_init.csrVW  = nullptr;
    cu_init.csrEW  = nullptr;

    Timer<DEVICE> TM;
    int is_rmat = 0;

    // Testing the scalablity of edge insertions and deletions for
    // batch sizes within the range of {1, 10, 100, .. 10^7}
    for (int batch_size :{1, 10, 100, 1000, 10000, 100000, 1000000, 10000000}) {
        // Running each experiment 5 times
        for (int i = 0; i < 5; i++) {
            cuStinger custing2(defaultInitAllocater, defaultUpdateAllocater);
            TM.start();
            custing2.initializeCuStinger(cu_init);
            TM.stop();

            std::cout << graph.name() << "," << graph.nV() << "," << graph.nE()
                      << "," << batch_size << "," << TM.duration() << flush;

            printcuStingerUtility(custing2, false);

            BatchUpdateData bud(batch_size,true);
            // Creating the batch update.
            if(is_rmat) {   // Using rmat graph generator.
                double a = 0.55, b = 0.15, c = 0.15, d = 0.25;
                dxor128_env_t env;// dxor128_seed(&env, 0);
                generateEdgeUpdatesRMAT(graph.nV(), batch_size, bud.getSrc(),bud.getDst(),a,b,c,d,&env);
            }
            else { // Using a uniform random graph generator.
                generateInsertBatch(bud.getSrc(), bud.getDst(), batch_size,
                                     graph);
            }

            BatchUpdate bu(bud);

            // custing2.checkDuplicateEdges();
            // custing2.verifyEdgeInsertions(bu);
            // cout << "######STARTING INSERTIONS######"<< endl;
            // Inserting the edges into the graph.
            length_t allocs;
            TM.start();
            custing2.edgeInsertions(bu, allocs);
            TM.stop();
            std::cout << "," << TM.duration() << "," << allocs;

            // custing2.verifyEdgeInsertions(bu);
            // cout << "The graphs are identical" << custing2.verifyEdgeInsertions(bu) << endl;//
            printcuStingerUtility(custing2, false);
            // custing2.checkDuplicateEdges();

            TM.start();
            custing2.edgeDeletions(bu); // Inserting the deletions into the graph.
            TM.stop();
            std::cout << "," << TM.duration();

            custing2.verifyEdgeDeletions(bu);
            printcuStingerUtility(custing2, false);
            std::cout << std::endl;
            custing2.freecuStinger();
        }
    }
}

/**
 * @brief Printer utility function that gets the percentage of utilized space in
 *        the graph.
 * @param[in] custing
 * @param[in] all_info it is prints the number of used edges and allocated edges
 */
void printcuStingerUtility(cuStinger custing, bool all_info) {
    auto used      = custing.getNumberEdgesUsed();
    auto allocated = custing.getNumberEdgesAllocated();
    if (all_info)
        std::cout << "," << used << "," << allocated;
    std::cout << "," << static_cast<float>(used) /static_cast<float>(allocated);
}
