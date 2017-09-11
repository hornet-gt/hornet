/**
 * @brief Sparse Matrix-Vector multiplication
 * @file
 */
#include "Static/SpMV/SpMV.cuh"
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
//#include <cuda_profiler_api.h> //--profile-from-start off

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornet_alg;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);
    auto h_vector = new int[graph.nV()];
    auto  h_value = new int[graph.nE()];
    std::fill(h_vector, h_vector + graph.nV(), 1);
    std::fill(h_value, h_value + graph.nE(), 1);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                           graph.out_edges_ptr());
    hornet_init.insertEdgeData(h_value);

    HornetGPU hornet_matrix(hornet_init);
    SpMV spmv(hornet_matrix, h_vector);
    
    Timer<DEVICE> TM;
    TM.start();

    spmv.run();

    TM.stop();
    TM.print("SpMV");

    auto is_correct = spmv.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    delete[] h_vector;
    delete[] h_value;
    return is_correct;
}
