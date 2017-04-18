///@file
#include <type_traits>

namespace cu_stinger {

__constant__ size_t     d_nV = 0;
__constant__ byte_t* d_ptrs[NUM_VTYPES];

} // namespace cu_stinger

#include "cuStingerTypes.cuh"

__device__ int aaa[10];

namespace cu_stinger {

__global__ void printKernel2() {
    for (id_t i = 0; i < d_nV; i++) {
        auto vertex = Vertex(i);
        //auto degree = vertex.degree();
        //auto field0 = vertex.field<0>();
        //printf("%d [%d]:\t", i, vertex.degree());

        for (degree_t j = 0; j < vertex.degree(); j++) {
            auto edge = vertex.edge(j);
            //auto weight = edge.weight();
            //auto  time1 = edge.time_stamp1();
            //auto field0 = edge.field<0>();
            //auto field1 = edge.field<1>();

            //printf("%d\t", edge.dst());
            aaa[j] = edge.dst();
        }
        //printf("\n");
    }
    //printf("\n");
    //from RAW:
    //
    //for (id_t i = 0; i < d_nV; i++) {
    //  for (degree_t j = 0; j < vertex.degrees(); j++) {
    //       auto edge = vertex.edge(i);
    //----------------------------------------------------
    //to PROPOSED:
    //
    //for (auto v : VertexSet) {
    //  for (auto edge : v) {
}

__global__ void printKernel() {
    degree_t* degrees = reinterpret_cast<degree_t*>(d_ptrs[0]);
    degree_t*  limits = reinterpret_cast<degree_t*>(d_ptrs[1]);
    id_t**      edges = reinterpret_cast<id_t**>(d_ptrs[2]);

    for (id_t i = 0; i < d_nV; i++) {
        printf("%d [%d, %d]:\t", i, degrees[i], limits[i]);
        for (degree_t j = 0; j < degrees[i]; j++)
            printf("%d\t", edges[i][j]);
        printf("\n");
    }
    printf("\n");
}

} // namespace cu_stinger
