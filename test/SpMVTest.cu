/**
 * @brief Sparse Matrix-Vector multiplication
 * @file
 */
#include "Static/SpMV/SpMV.cuh"
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
//#include <cuda_profiler_api.h> //--profile-from-start off
#include <cub.cuh>

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);
    auto h_vector = new int[graph.nV()];
    auto  h_value = new int[graph.nE()];
    std::fill(h_vector, h_vector + graph.nV(), 1);
    std::fill(h_value, h_value + graph.nE(), 1);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                           graph.out_edges_ptr());
    hornet_init.insertEdgeData(h_value);

    HornetGraph hornet_matrix(hornet_init);
    SpMV spmv(hornet_matrix, h_vector);

    Timer<DEVICE> TM;
    TM.start();

    spmv.run();

    TM.stop();
    TM.print("SpMV");

    auto is_correct = spmv.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");

    TM.start();

    /*int* d_row_offsets = const_cast<int*>(graph.out_offsets_ptr());
    int* d_column_indices = const_cast<int*>(graph.out_edges_ptr());
    float* d_values  = (float*) h_value;
    float* d_vector_x = (float*) h_vector;
    int num_rows = graph.nV();
    int num_cols = graph.nV();
    int num_nonzeros = graph.nE();
    float* d_vector_y;
    cuMalloc(d_vector_y, graph.nV());

    void*    d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                           d_row_offsets, d_column_indices, d_vector_x,
                           d_vector_y, num_rows, num_cols, num_nonzeros);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                           d_row_offsets, d_column_indices, d_vector_x,
                           d_vector_y, num_rows, num_cols, num_nonzeros);
    TM.stop();
    TM.print("Cub SpMV");*/

    delete[] h_vector;
    delete[] h_value;
    return is_correct;
}
