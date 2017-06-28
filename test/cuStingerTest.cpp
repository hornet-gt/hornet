///@files
#include "Core/cuStinger.hpp"

#include "GraphIO/GraphStd.hpp"        //GraphStd
#include "Util/BatchFunctions.hpp"        //GraphStd

//#include "Util/Parameters.hpp"         //Param
#include "Support/Host/FileUtil.hpp"   //xlib::extract_filepath_noextension
#include "Support/Device/CudaUtil.cuh" //xlib::deviceInfo
#include "Support/Host/Timer.hpp"      //Timer<HOST>
#include <algorithm>                   //std:.generate
#include <chrono>                      //std::chrono
#include <random>                      //std::mt19937_64

using namespace custinger;
using namespace timer;

void exec(int argc, char* argv[]);

/**
 * @brief Example tester for cuSTINGER.
 * Loads an input graph, creates a batches of edges, inserts them into the
 * graph, and then removes them from the graph.
 */
int main(int argc, char* argv[]) {
    exec(argc, argv);
    cudaDeviceReset();
}

void exec(int argc, char* argv[]) {
    xlib::deviceInfo();
    //Param param(argc, argv);

    graph::GraphStd<custinger::vid_t, custinger::eoff_t> graph(graph::Structure::UNDIRECTED);
    graph.read(argv[1], graph::Property(static_cast<graph::Property::Enum>(6)));
    graph.print();
    //graph.print_raw();

    //if (param.binary)
    //    graph.toBinary(xlib::extract_filepath_noextension(argv[1]) + ".bin");
    //--------------------------------------------------------------------------

    auto seed = std::chrono::high_resolution_clock::now()
                .time_since_epoch().count();
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<uint64_t>      int_dist(-10000, 10000);
    std::uniform_int_distribution<unsigned char> char_dist(0, 255);
    std::uniform_real_distribution<float>        float_dist(-100.0f, 100.0f);

    /*auto     labels = new unsigned char[graph.nV()];
    auto time_stamp = new uint64_t[graph.nE()];


    std::generate(labels, labels + graph.nV(), [&]{ return char_dist(gen); });
    std::generate(weights, weights + graph.nE(),
                  [&]{ return float_dist(gen); });
    std::generate(time_stamp, time_stamp + graph.nE(),
                  [&]{ return int_dist(gen); });*/
    //--------------------------------------------------------------------------
    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets(),
                                 graph.out_edges());

    auto weights = new int[graph.nE()]();
    weights[0] = 0;
    weights[1] = 1;
    weights[2] = 2;
    weights[3] = 3;
    //custinger_init.insertVertexData(labels);
    custinger_init.insertEdgeData(weights);

    cuStinger custiger_graph(custinger_init);
    //custiger_graph.check_consistency(custinger_init);

    //delete[] labels;
    //delete[] time_stamp;
    delete[] weights;

    custiger_graph.print();
    //--------------------------------------------------------------------------
    /*int batch_size = 10;
    auto batch_src = new vid_t[batch_size];
    auto batch_dst = new vid_t[batch_size];
    generateInsertBatch(batch_src, batch_dst, batch_size, graph);*/
                        //batch_property::PRINT);

    vid_t  batch_src[] = { 0, 2 };
    vid_t  batch_dst[] = { 2, 3 };
    int batch_size = sizeof(batch_src) / sizeof(vid_t);

    BatchInit batch_init(batch_src, batch_dst, batch_size);
    //BatchUpdate batch_update(batch_init);
    BatchUpdate batch_update(custiger_graph.nV());


    batch_update.insert(batch_init);
    //custiger_graph.insertEdgeBatch(batch_update);
    custiger_graph.edgeDeletionsSorted(batch_update);

    std::cout << "\n\n";
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
