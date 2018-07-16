
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


// void ClusteringCoefficient::copyTCToHost(triangle_t* h_tcs) {
//     gpu::copyToHost(triPerVertex, hornet.nV(), h_tcs);
// }

// triangle_t ClusteringCoefficient::countTriangles(){

//     triangle_t* h_triPerVertex;
//     host::allocate(h_triPerVertex, hornet.nV());
//     gpu::copyToHost(triPerVertex, hornet.nV(), h_triPerVertex);
//     triangle_t sum=0;
//     for(int i=0; i<hornet.nV(); i++){
//         // printf("%d %ld\n", i,outputArray[i]);
//         sum+=h_triPerVertex[i];
//     }
//     free(h_triPerVertex);
//     //triangle_t sum=gpu::reduce(hd_triangleData().triPerVertex, hd_triangleData().nv+1);

//     return sum;
// }

struct OPERATOR_LocalClusteringCoefficients {
    triangle_t *d_triPerVertex;
    float      *d_ccLocal;

    OPERATOR (Vertex &vertex) {
        degree_t deg = vertex.degree();
        d_ccLocal[vertex.id()] = 0;

        if(deg>1){
            d_ccLocal[vertex.id()] = (float)d_triPerVertex[vertex.id()]/(float)(deg*(deg-1));
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

    // float sum=gpu::reduce(d_ccLocal, hornet.nV());
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

} // namespace hornets_nest
