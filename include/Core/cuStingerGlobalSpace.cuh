
namespace cu_stinger {

__constant__ size_t     d_nV = 0;
__constant__ vertex_t* d_nodes = nullptr;

__global__ void printKernel() {
    degree_t* degrees = reinterpret_cast<degree_t*>(d_nodes);
    degree_t*  limits = degrees + d_nV;
    id_t**      edges = reinterpret_cast<id_t**>(limits + d_nV);

    for (id_t i = 0; i < d_nV; i++) {
        printf("%d [%d, %d]:\t", i, degrees[i], limits[i]);
        for (degree_t j = 0; j < degrees[i]; j++)
            printf("%d\t", edges[i][j]);
        printf("\n");
    }
    printf("\n");
}

} // namespace cu_stinger
