///@files
#include "Core/cuStinger.hpp"

#include "GraphIO/GraphStd.hpp" //GraphStd
#include "Util/Parameters.hpp"  //Param
#include "Support/FileUtil.hpp" //xlib::extract_filepath_noextension
#include "Support/CudaUtil.cuh" //xlib::deviceInfo
#include "Support/Timer.cuh"    //Timer<HOST>
#include <algorithm>
#include <chrono>
#include <random>

using namespace cu_stinger;
using namespace timer;

/**
 * @brief Example tester for cuSTINGER.
 * Loads an input graph, creates a batches of edges, inserts them into the
 * graph, and then removes them from the graph.
 */
int main(int argc, char* argv[]) {
    xlib::deviceInfo();
    Param param(argc, argv);

    graph::GraphStd<id_t, off_t> graph;
    graph.read(argv[1]);

    if (param.binary)
        graph.toBinary(xlib::extract_filepath_noextension(argv[1]) + ".bin");

    cuStinger custiger_graph(graph.nV(), graph.nE(), graph.out_offsets_array(),
                             graph.out_edges_array());
    //--------------------------------------------------------------------------

    auto seed = std::chrono::high_resolution_clock::now()
                .time_since_epoch().count();
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<uint64_t>      int_dist(-10000, 10000);
    std::uniform_int_distribution<unsigned char> char_dist(0, 255);
    std::uniform_real_distribution<float>        float_dist(-100.0f, 100.0f);

    auto     labels = new unsigned char[graph.nV()];
    auto time_stamp = new uint64_t[graph.nE()];
    auto    weights = new float[graph.nE()];

    std::generate(labels, labels + graph.nV(), [&]{ return char_dist(gen); });
    std::generate(weights, weights + graph.nE(),
                  [&]{ return float_dist(gen); });
    std::generate(time_stamp, time_stamp + graph.nE(),
                  [&]{ return int_dist(gen); });

    //custiger_graph.insertVertexData(labels);
    //custiger_graph.insertEdgeData(time_stamp, weights);
    custiger_graph.initialize();
    //--------------------------------------------------------------------------

    delete[] labels;
    delete[] time_stamp;
    delete[] weights;

    custiger_graph.print();

    //Timer<DEVICE> TM;

    // Testing the scalablity of edge insertions and deletions for
    // batch sizes within the range of {1, 10, 100, .. 10^7}
    /*for (int batch_size :{1, 10, 100, 1000, 10000, 100000, 1000000, 10000000}) {
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
    }*/
}
