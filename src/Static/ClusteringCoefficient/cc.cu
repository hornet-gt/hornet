
#include <cuda.h>
#include <cuda_runtime.h>

#include "Static/ClusteringCoefficient/cc.cuh"
#include "Static/TriangleCounting/triangle2.cuh"

namespace hornets_nest {

ClusteringCoefficient::ClusteringCoefficient(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet)                                      
{                                      
    tri = new TriangleCounting2(hornet);   
}

ClusteringCoefficient::~ClusteringCoefficient(){
    tri->release();
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


void ClusteringCoefficient::reset(){
    tri->reset();
}

void ClusteringCoefficient::run(){
    tri->run();
}


void ClusteringCoefficient::release(){
    gpu::free(ccLocal);
    ccLocal = nullptr;
}

void ClusteringCoefficient::init(){
    //printf("Inside init. Printing hornet.nV(): %d\n", hornet.nV());
    gpu::allocate(ccLocal, hornet.nV());
    reset();
}

} // namespace hornets_nest
