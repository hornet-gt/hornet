/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>       
 *   ogreen@nvidia.com
 *   @author Muhammad Osama Sakhi                                           <br>
 *   Georgia Institute of Technology                                        <br>       
 * @date July, 2018
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
 */


#include "Static/BetweennessCentrality/bc.cuh"

#include "bcOperators.cuh"

using length_t = int;
using namespace std;
namespace hornets_nest {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

BCCentrality::BCCentrality(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet),
                                       load_balancing(hornet)
{
    hd_BCData().currLevel=0;
    cout << "hornet.nV   " << hornet.nV() << endl;

    host::allocate(hd_BCData().depth_indices, hornet.nV());
    gpu::allocate(hd_BCData().d, hornet.nV());

    gpu::allocate(hd_BCData().sigma, hornet.nV());
    gpu::allocate(hd_BCData().delta, hornet.nV());
    gpu::allocate(hd_BCData().bc, hornet.nV());
    hd_BCData().queue.initialize(hornet);

    reset();
}

BCCentrality::~BCCentrality() {
    release();
}

void BCCentrality::reset() {
    hd_BCData().currLevel=0;

    forAllnumV(hornet, InitBC { hd_BCData });
    forAllnumV(hornet, InitOneTree { hd_BCData });
    hd_BCData.sync();
}

void BCCentrality::release(){
    host::free(hd_BCData().depth_indices);

    gpu::free(hd_BCData().d);
    gpu::free(hd_BCData().sigma);
    gpu::free(hd_BCData().delta);
    gpu::free(hd_BCData().bc);
}

void BCCentrality::setRoot(vid_t root_){
    hd_BCData().root=root_;
}

void BCCentrality::run() {


    // Initialization
    hd_BCData().currLevel=0;
    forAllnumV(hornet, InitOneTree { hd_BCData });
    vid_t root = hd_BCData().root;

    hd_BCData().queue.insert(root);                   // insert source in the frontier
    gpu::memsetZero(hd_BCData().d + root);

    // Regular BFS
    hd_BCData().depth_indices[0]=1;

    vid_t* depArray;
    gpu::allocate(depArray, hornet.nV());


    cudaMemcpy(depArray,hd_BCData().queue.device_input_ptr(),sizeof(vid_t)*hd_BCData().queue.size(),cudaMemcpyDeviceToDevice);
    // cout << hd_BCData().depth_indices[0] <<  " " <<  1 << endl;

    while (hd_BCData().queue.size() > 0) {

        if(hd_BCData().currLevel>0) {
            length_t prevLength = hd_BCData().depth_indices[hd_BCData().currLevel] - hd_BCData().depth_indices[hd_BCData().currLevel-1];

            cudaMemcpy(depArray+hd_BCData().depth_indices[hd_BCData().currLevel-1],hd_BCData().queue.device_input_ptr(),
                sizeof(vid_t)*(prevLength) ,cudaMemcpyDeviceToDevice);
            // cout << hd_BCData().depth_indices[hd_BCData().currLevel-1] <<  " " <<  prevLength << endl;
        }


        hd_BCData().depth_indices[hd_BCData().currLevel+1]=
                       hd_BCData().depth_indices[hd_BCData().currLevel]+hd_BCData().queue.size();

        forAllEdges(hornet, hd_BCData().queue, BC_BFSTopDown { hd_BCData }, load_balancing);
        // hd_BCData.sync();
        hd_BCData().currLevel++;
        // cout << "Depth in the Forward-BFS " <<  hd_BCData().queue.size() << endl;

        hd_BCData().queue.swap();
    }

    hd_BCData().currLevel -= 2;

    // Reverse BFS - Dependency accumulation
    while (hd_BCData().currLevel>=0) {
        // int2 pos={hd_BCData().depth_indices[hd_BCData().currLevel],
        //     hd_BCData().depth_indices[hd_BCData().currLevel + 1]};
        length_t prevLength = hd_BCData().depth_indices[hd_BCData().currLevel+1] - hd_BCData().depth_indices[hd_BCData().currLevel];


        // cout << hd_BCData().depth_indices[hd_BCData().currLevel] <<  " " <<  prevLength << endl;



        // forAllEdges(hornet, hd_BCData().queue, BC_DepAccumulation { hd_BCData }, load_balancing);
        forAllEdges(hornet, depArray+hd_BCData().depth_indices[hd_BCData().currLevel],prevLength, BC_DepAccumulation { hd_BCData }, load_balancing);

        forAllnumV(hornet, IncrementBC { hd_BCData });

        hd_BCData().currLevel--;
    }
    gpu::free(depArray);

}


// int BCCentrality::getDepth() {
//     return hd_BCData();
// }

bool BCCentrality::validate() {
    return true;
}

} // namespace hornets_nest
