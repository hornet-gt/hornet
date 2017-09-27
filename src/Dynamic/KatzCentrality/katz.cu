/**
 * @internal
 * @author Oded Green                                                  <br>
 *         Georgia Institute of Technology, Computational Science and Engineering                   <br>
 *         ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
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
 * @file
 */
#include "Dynamic/KatzCentrality/katz.cuh"

namespace hornet_alg {

katzCentralityDynamic::katzCentralityDynamic(HornetGPU& hornet) :
                                       StaticAlgorithm(hornet),
                                       load_balacing(hornet),
                                       hd_katzdata.active_queue(hornet) {

    std::cout << "Oded remember to take care of memory de-allocation\n"
              << "Oded need to figure out correct API for dynamic graph"
              << "algorithms\n"
              << "Dynamic katz centrality algorithm needs to get both the"
              << "original graph and the inverted graph for directed graphs"
              << std::endl;
}

katzCentralityDynamic::~katzCentralityDynamic() {
    release();
}

void katzCentralityDynamic::setInitParametersUndirected(int maxIteration_, int K_,degree_t maxDegree_){
    kcStatic.setInitParameters(maxIteration_,K_,maxDegree_,false);
    is_directed = false;
}
void katzCentralityDynamic::setInitParametersDirected(int maxIteration_, int K_,degree_t maxDegree_, cuStinger* invertedGraph__){
    kcStatic.setInitParameters(maxIteration_, K_, maxDegree_, false);
    invertedGraph = invertedGraph__;
    is_directed   = true;
}

void katzCentralityDynamic::init(){
    // Initializing the static graph KatzCentrality data structure
    kcStatic.init();

    host::copyToHost(kcStatic.getHostKatzData(), 1, &hd_katzdata,);
    gpu::copyToDevice(kcStatic.getDeviceKatzData(), 1, deviceKatzData);

    gpu::allocate(hd_katzdata.new_paths_curr, hornet.nV());
    gpu::allocate(hd_katzdata.new_paths_prev, hornet.nV());
    gpu::allocate(hd_katzdata.active,         hornet.nV());
}


void katzCentralityDynamic::runStatic() {
    // Executing the static graph algorithm
    kcStatic.reset();
    kcStatic.run(hornet);

    host::copyToHost(kcStatic.getHostKatzData(), 1, &hd_katzdata,);
    gpu::copyToDevice(kcStatic.getDeviceKatzData(), 1, deviceKatzData);

    hd_katzdata.iteration_static = hd_katzdata.iteration;

    // Initializing the fields of the dynamic graph algorithm
    forAllVertices<initStreaming>(hornet, InitStreaming { deviceKatzData } );
}

void katzCentralityDynamic::release(){
    gpu::free(hd_katzdata.new_paths_curr);
    gpu::free(hd_katzdata.new_paths_prev);
    gpu::free(hd_katzdata.active);
    gpu::free(deviceKatzData);
}

//==============================================================================

int katzCentralityDynamic::get_iteration_count(){
    return hd_katzdata.iteration;
}

void katzCentralityDynamic::batchUpdateInsertion(BatchUpdate &batch_update) {
    processUpdate(batch_update, true);
}

void katzCentralityDynamic::batchUpdateDeleted(BatchUpdate &batch_update) {
    processUpdate(batch_update, false);
}

void katzCentralityDynamic::processUpdate(BatchUpdate& batch_update,
                                          bool is_insert) {
    // Resetting the queue of the active vertices.
    hd_katzdata.active_queue.clear();
    hd_katzdata.iteration = 1;

    // Initialization of insertions or deletions is slightly different.
    if (is_insert) {
        allEinA_TraverseEdges(hornet,
                              setupInsertions { deviceKatzData, batch_update });
    }
    else {
        allEinA_TraverseEdges(hornet,
                              SetupDeletions { deviceKatzData, batch_update } );
    }
    syncHostWithDevice();

    hd_katzdata.iteration = 2;
    hd_katzdata.num_active   = hd_katzdata.active_queue.getQueueEnd();

    while (hd_katzdata.iteration < hd_katzdata.maxIteration &&
           hd_katzdata.iteration < hd_katzdata.iteration_static) {
        hd_katzdata.alphaI = std::pow(hd_katzdata.alpha, hd_katzdata.iteration);

         /*is the same??*/
        forAll(hornet, hd_katzdata.active_queue, hd_katzdata.num_active
                    InitActiveNewPaths { deviceKatzData };

        // Undirected graphs and directed graphs need to be dealt with differently.
        if (!is_directed) {
            forAllEdges(hornet, hd_katzdata.active_queue,
                        FindNextActive {deviceKatzData}, load_balacing );
            syncHostWithDevice(); // Syncing queue info

            forAllEdges(hornet, hd_katzdata.active_queue,
                        UpdateActiveNewPaths { deviceKatzData },
                        load_balacing );
        }
        else {
            forAllEdges(*invertedGraph, hd_katzdata.active_queue,
                        FindNextActive {deviceKatzData}, load_balacing);
            syncHostWithDevice(); // Syncing queue info
            forAllEdges(*invertedGraph, hd_katzdata.active_queue,
                        UpdateActiveNewPaths {deviceKatzData}, load_balacing);
        }
        syncHostWithDevice(); // Syncing queue info

        // Checking if we are dealing with a batch of insertions or deletions.
        if (is_insert)
            allEinA_TraverseEdges<updateNewPathsBatchInsert>(hornet, deviceKatzData,batch_update);
        }else{
            allEinA_TraverseEdges<updateNewPathsBatchDelete>(hornet, deviceKatzData,batch_update);
        }
        syncHostWithDevice();

        hd_katzdata.num_active = hd_katzdata.active_queue.getQueueEnd();
        allVinA_TraverseVertices<updatePrevWithCurr>(hornet, deviceKatzData, hd_katzdata.active_queue.getQueue(), hd_katzdata.num_active);
        syncHostWithDevice();

        hd_katzdata.iteration++;
    }

    if (hd_katzdata.iteration > 2) {
        allVinA_TraverseVertices<updateLastIteration>(hornet, deviceKatzData, hd_katzdata.active_queue.getQueue(), hd_katzdata.num_active);
        syncHostWithDevice();
    }
    // Resetting the fields of the dynamic graph algorithm for all the vertices that were active
    //hd_katzdata.num_active ??
    forAllVertices(hornet, hd_katzdata.active_queue,
                   InitStreaming {deviceKatzData});
}

}// cuStingerAlgs namespace
