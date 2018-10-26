
#include <cuda.h>
#include <cuda_runtime.h>

#include "Static/ClusteringCoefficient/cc.cuh"
#include "Static/TriangleCounting/triangle2.cuh"

using namespace xlib;

namespace hornets_nest {

ClusteringCoefficient::ClusteringCoefficient(HornetGraph& hornet) :
                                        TriangleCounting2(hornet)
                                       // StaticAlgorithm(hornet)                                      
{
    // tri = new TriangleCounting2(hornet);   
}

ClusteringCoefficient::~ClusteringCoefficient(){
    // tri->release();
    TriangleCounting2::release();
    release();
}


struct OPERATOR_LocalClusteringCoefficients {
    triangle_t *d_triPerVertex;
    clusterCoeff_t      *d_ccLocal;

    OPERATOR (Vertex &vertex) {
        degree_t deg = vertex.degree();
        d_ccLocal[vertex.id()] = 0;

        if(deg>1){
            d_ccLocal[vertex.id()] = (clusterCoeff_t)d_triPerVertex[vertex.id()]/(clusterCoeff_t)(deg*(deg-1));
        }
    }
};


void ClusteringCoefficient::reset(){
    TriangleCounting2::reset();
}

void ClusteringCoefficient::run(){
    TriangleCounting2::run();
    forAllVertices(hornet, OPERATOR_LocalClusteringCoefficients { triPerVertex,d_ccLocal }); 


    // int* d_ccLocalInt;
    // int sumInt=gpu::reduce(d_ccLocalInt, 10);

    // clusterCoeff_t sum=gpu::reduce(d_ccLocal, hornet.nV());
}


void ClusteringCoefficient::release(){
    gpu::free(d_ccLocal);
    d_ccLocal = nullptr;
}

void ClusteringCoefficient::init(){
    //printf("Inside init. Printing hornet.nV(): %d\n", hornet.nV());
    gpu::allocate(d_ccLocal, hornet.nV());
    TriangleCounting2::init();
    reset();
}

void ClusteringCoefficient::copyLocalClusCoeffToHost(clusterCoeff_t* h_tcs){
    gpu::copyToHost(d_ccLocal, hornet.nV(), h_tcs);

}

} // namespace hornets_nest
