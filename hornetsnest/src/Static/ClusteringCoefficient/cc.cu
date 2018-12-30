
/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>       
 *   ogreen@nvidia.com
 * @date October, 2018
 *
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 * 
 * Please cite:
 * * J. Fox, O. Green, K. Gabert, X. An, D. Bader, “Fast and Adaptive List Intersections on the GPU”, 
 * IEEE High Performance Extreme Computing Conference (HPEC), 
 * Waltham, Massachusetts, 2018
 * * O. Green, J. Fox, A. Tripathy, A. Watkins, K. Gabert, E. Kim, X. An, K. Aatish, D. Bader, 
 * “Logarithmic Radix Binning and Vectorized Triangle Counting”, 
 * IEEE High Performance Extreme Computing Conference (HPEC), 
 * Waltham, Massachusetts, 2018
 * * O. Green, P. Yalamanchili ,L.M. Munguia, “Fast Triangle Counting on GPU”, 
 * Irregular Applications: Architectures and Algorithms (IA3), 
 * New Orleans, Louisiana, 2014
 * 
 */


#include "Static/ClusteringCoefficient/cc.cuh"
#include "Static/TriangleCounting/triangle2.cuh"

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

    int _num_items = hornet.nV();

    byte_t*  _d_temp_storage     { nullptr };
    size_t _temp_storage_bytes { 0 };
    cub::DeviceReduce::Sum(_d_temp_storage, _temp_storage_bytes,d_ccLocal, d_ccGlobal, _num_items); // Allocating storage needed by CUB for the reduce
    gpu::allocate(_d_temp_storage, _temp_storage_bytes);
    cub::DeviceReduce::Sum(_d_temp_storage, _temp_storage_bytes, d_ccLocal, d_ccGlobal, _num_items);

    gpu::copyToHost(d_ccGlobal, 1, &h_ccGlobal);
    gpu::free(_d_temp_storage);

    std::cout << "Global CC " << h_ccGlobal/hornet.nV() << std::endl;
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
