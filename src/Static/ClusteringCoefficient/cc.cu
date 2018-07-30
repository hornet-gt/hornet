
#include <cuda.h>
#include <cuda_runtime.h>

#include "Static/ClusteringCoefficient/cc.cuh"
#include "Static/TriangleCounting/triangle2.cuh"

// #include <Device/Util/SafeCudaAPI.cuh>
// #include <Device/Primitives/CubWrapper.cuh>

#include "Core/StandardAPI.hpp"

using namespace xlib;
using namespace gpu;

namespace hornets_nest {

ClusteringCoefficient::ClusteringCoefficient(HornetGraph& hornet) :
                                        TriangleCounting2(hornet)
                                       // StaticAlgorithm(hornet)                                      
{
}

ClusteringCoefficient::~ClusteringCoefficient(){
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
#include <cub/cub.cuh>
void ClusteringCoefficient::run(){
    TriangleCounting2::run();
    forAllVertices(hornet, OPERATOR_LocalClusteringCoefficients { triPerVertex,d_ccLocal }); 


    // int* d_ccLocalInt;
    // int sumInt=gpu::reduce<clusterCoeff_t>(d_ccLocal, size_t(hornet.nV()));

    // xlib::CubReduce<clusterCoeff_t> red(d_ccLocal, hornet.nV());
    // gpu::reduce

    int _num_items = hornet.nV();

    void*  _d_temp_storage     { nullptr };
    size_t _temp_storage_bytes { 0 };
    cub::DeviceReduce::Sum(_d_temp_storage, _temp_storage_bytes,d_ccLocal, d_ccGlobal, _num_items); // Allocating storage needed by CUB for the reduce
    cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
    cub::DeviceReduce::Sum(_d_temp_storage, _temp_storage_bytes, d_ccLocal, d_ccGlobal, _num_items);

    gpu::copyToHost(d_ccGlobal, 1, &h_ccGlobal);
    gpu::free(_d_temp_storage);

    std::cout << "Global CC " << h_ccGlobal/hornet.nV() << std::endl;
    // clusterCoeff_t sum=gpu::reduce(d_ccLocal, hornet.nV());
}


void ClusteringCoefficient::release(){
    gpu::free(d_ccLocal);
    gpu::free(d_ccGlobal);
    d_ccLocal = nullptr;
}

void ClusteringCoefficient::init(){
    //printf("Inside init. Printing hornet.nV(): %d\n", hornet.nV());
    gpu::allocate(d_ccLocal, hornet.nV());
    gpu::allocate(d_ccGlobal, 1);

    TriangleCounting2::init();
    reset();
}

void ClusteringCoefficient::copyLocalClusCoeffToHost(clusterCoeff_t* h_tcs){
    gpu::copyToHost(d_ccLocal, hornet.nV(), h_tcs);
}

} // namespace hornets_nest
