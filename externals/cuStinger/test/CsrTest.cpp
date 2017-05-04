///@files
#include "Core/cuStinger.hpp"
#include "Csr/Csr.hpp"

#include "GraphIO/GraphStd.hpp"        //GraphStd
#include "Util/Parameters.hpp"         //Param
#include "Support/Host/FileUtil.hpp"   //xlib::extract_filepath_noextension
#include "Support/Device/CudaUtil.cuh" //xlib::deviceInfo
#include "Support/Host/Timer.hpp"      //Timer<HOST>
#include <algorithm>                   //std:.generate
#include <chrono>                      //std::chrono
#include <random>                      //std::mt19937_64

using namespace custinger;
using namespace csr;
using namespace timer;

/**
 * @brief Example tester for cuSTINGER.
 * Loads an input graph, creates a batches of edges, inserts them into the
 * graph, and then removes them from the graph.
 */
int main(int argc, char* argv[]) {
    xlib::deviceInfo();
    Param param(argc, argv);

    graph::GraphStd<custinger::id_t, custinger::off_t> graph;
    graph.read(argv[1]);
    graph.print_raw();

    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets(),
                                 graph.out_edges());

    Csr csr_graph(custinger_init);
    csr_graph.print();
}
