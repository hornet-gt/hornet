/**
 * @internal
 * @author Euna Kim                                               <br>
 *         Georgia Institute of Technology, Computational Science and Engineering                   <br>
 *         euna.kim@gatech.edu
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
#include "Dynamic/PageRank/PageRank.cuh"
#include "PageRankOperators.cuh"

namespace hornets_nest {
//namespace hornet_alg {
//#if 0
//PageRankDynamic::PageRankDynamic(HornetGPU& hornet, //HornetGraph& hornet,
//                                HornetGPU& inverted_graph, //HornetGraph& inverted_graph,
//                                int  iteration_max,
//                                pr_t threshold,
//                                pr_t damp) :
//                         StaticAlgorithm(hornet),
//                         load_balancing(hornet), //load_balancing(hornet),
//                         inverted_graph(inverted_graph),
//                         is_directed(false),
//                         pr_static(hornet, iteration_max, 
//                                threshold, damp, false) {
//#else
//PageRankDynamic::PageRankDynamic(HornetGPU& hornet, //HornetGraph& hornet,
//                                HornetGPU& inverted_graph, //HornetGraph& inverted_graph,
//                                int  iteration_max,
//                                pr_t threshold,
//                                pr_t damp) :
//                         StaticAlgorithm(hornet),
//                         load_balancing(hornet), //load_balancing(hornet),
//                         inverted_graph(inverted_graph),
//                         is_directed(false),
//                         pr_static(hornet, iteration_max, 
//                                threshold, damp) { //, false) {
//#endif
//    printf("------------- dynamic pr 1");
//
//    //hd_prdata().active_queue.initialize(hornet);
//    hd_prdata().queue1.initialize(hornet);//initialize(hornet);
//    hd_prdata().queue2.initialize(hornet);//initialize(hornet);
//    hd_prdata().queue3.initialize(hornet);//initialize(hornet);
//    hd_prdata().queueDlt.initialize(hornet);//initialize(hornet);
//
//    gpu::allocate(hd_prdata().visited,    hornet.nV());
//    gpu::allocate(hd_prdata().visitedDlt, hornet.nV());
//    gpu::allocate(hd_prdata().usedOld,    hornet.nV());
//    gpu::allocate(hd_prdata().diffPR,     hornet.nV());
//    gpu::allocate(hd_prdata().delta,      hornet.nV());
//
//    hd_prdata = pr_static.pr_data();
//
//    std::cout << "Oded remember to take care of memory de-allocation\n"
//              << "Oded need to figure out correct API for dynamic graph"
//              << "algorithms\n"
//              << "Dynamic PageRank algorithm needs to get both the"
//              << "original graph and the inverted graph for directed graphs"
//              << std::endl;
//}
PageRankDynamic::PageRankDynamic(HornetGPU& hornet, ////HornetGraph& hornet,
                                 int  iteration_max,
                                 pr_t threshold,
                                 pr_t damp) :
                        StaticAlgorithm(hornet),
                        load_balancing(hornet), //load_balancing(hornet),
                        is_directed(true),
                        pr_static(hornet, iteration_max, 
                                 threshold, damp) { //, true) {

    printf("------------- dynamic pr 2");
    hd_prdata = pr_static.pr_data(); //!!!! ERROR!!!!
    printf("------------- dynamic pr 2");
    //hd_prdata().active_queue.initialize(hornet);
    hd_prdata().queue1.initialize(hornet);//initialize(hornet);
    hd_prdata().queue2.initialize(hornet);//initialize(hornet);
    hd_prdata().queue3.initialize(hornet);//initialize(hornet);
    hd_prdata().queueDlt.initialize(hornet);//initialize(hornet);

    printf("------------- dynamic pr 2");
    gpu::allocate(hd_prdata().visited,    hornet.nV());
    gpu::allocate(hd_prdata().visitedDlt, hornet.nV());
    gpu::allocate(hd_prdata().usedOld,    hornet.nV());
    gpu::allocate(hd_prdata().diffPR,     hornet.nV());
    gpu::allocate(hd_prdata().delta,      hornet.nV());

    //printf("------------- dynamic pr 2");
    std::cerr<<"------------- dynamic pr 2";

    //std::cout << "Oded remember to take care of memory de-allocation\n"
    //          << "Oded need to figure out correct API for dynamic graph"
    //          << "algorithms\n"
    //          << "Dynamic PageRank algorithm needs to get both the"
    //          << "original graph and the inverted graph for directed graphs"
    //          << std::endl;
}

PageRankDynamic::~PageRankDynamic() {
    release();
}

void PageRankDynamic::run() {

    //printf("run_static: 11111\n");

    // Executing the static graph algorithm
    pr_static.reset();
    pr_static.run();

    //hd_prdata().iteration_static = hd_prdata().iteration;
    hd_prdata().iteration = hd_prdata().iteration;

    // Initializing the fields of the dynamic graph algorithm
    forAllnumV(hornet, InitStreaming { hd_prdata });
}

void PageRankDynamic::release(){
    gpu::free(hd_prdata().visited);
    gpu::free(hd_prdata().visitedDlt);
    gpu::free(hd_prdata().usedOld);
    gpu::free(hd_prdata().diffPR);
    gpu::free(hd_prdata().delta);
}

bool PageRankDynamic::validate() {
	return true;//?????????
}

//==============================================================================
void PageRankDynamic::reset(){
    hd_prdata().iteration = 0;
    hd_prdata().queue1.clear();
    hd_prdata().queue2.clear();
    hd_prdata().queue3.clear();
    hd_prdata().queueDlt.clear();
}

void PageRankDynamic::processUpdate(BatchUpdate& batch_update,
                                    bool is_insert) {
    //printf("--(DynamicPR)processUpdate\n");
//---------------------------------------------------------------------        
//        generateBatch(graph, batch_size, batch_src, batch_dst,
//                      BatchGenType::INSERT, UNIQUE); //| PRINT
          vid_t batch_src[] = { 0, 0, 2 };
          vid_t batch_dst[] = { 2, 6, 7 };
          auto batch_size = 3;
//        gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);
//---------------------------------------------------------------------            

    // Resetting the queue of the active vertices.
    //hd_prdata().active_queue.clear();
    hd_prdata().iteration = 1;
    //hd_prdata().iteration = 0;
    hd_prdata().queue1.clear();
    hd_prdata().queue2.clear();
    hd_prdata().queue3.clear();
    hd_prdata().queueDlt.clear();

    //insert vertices in updating list in Q1
    for(auto i=0; i<batch_size; i++) {
        hd_prdata().queue1.insert(batch_src[i]);
        hd_prdata().queue1.insert(batch_dst[i]);
        //cout << "queueu1 enqueue [" << i << "]: " << edgeSrc[i] << endl;
    }  


//TO DO:: usedOld
//copyArrayHostToDevice((length_t*)len,hostPRData.usedOld,hostPRData.nv,sizeof(length_t));

//(update)--------------------------
    // Initialization of insertions or deletions is slightly different.
    if (is_insert) {
        //forAllEdges(hornet, batch_update, SetupInsertions { hd_prdata } );
        //forAllEdges(hornet, batch_update, RecomputeInsertionContriUndirected { hd_prdata } );
        forAllEdges(hornet, batch_update, RecomputeContri { hd_prdata } );
    } else {
        //forAllEdges(hornet, batch_update, SetupDeletions { hd_prdata } );
        forAllEdges(hornet, batch_update, RecomputeContri { hd_prdata } );
        //forAllEdges(hornet, batch_update, RecomputeDeletionContriUndirected { hd_prdata } );
    }
    hd_prdata.sync();

    //OK:: clear --> InitStreaming

    hd_prdata.sync(); // Syncing queue info
    //move dlt
    forAll(hd_prdata().queueDlt, //hd_prdata().num_active
            UpdateDltMove { hd_prdata });
    hd_prdata.sync(); // Syncing queue info
    //update contri --> curr_pr
    forAll(hd_prdata().queue2, //hd_prdata().num_active
            UpdateContriCopy { hd_prdata });
    hd_prdata.sync(); // Syncing queue info
    //move dlt
    forAll(hd_prdata().queueDlt, //hd_prdata().num_active
            UpdateDltCopy { hd_prdata });
    hd_prdata.sync(); // Syncing queue info

//(propagation)--------------------
    hd_prdata().iteration = 2;
    int i = 0;
    while(hd_prdata().queue2.size()>0)
    {
        //update contri --> curr_pr
        //forAll(hd_prdata().queue2, //hd_prdata().num_active
        //        UpdateContri { hd_prdata });
        forAllEdges(hornet, hd_prdata().queue2, 
                    UpdateContri { hd_prdata }, load_balancing);
        hd_prdata.sync(); // Syncing queue info
        //move dlt
        forAll(hd_prdata().queueDlt, //hd_prdata().num_active
                UpdateDltMove { hd_prdata });
        hd_prdata.sync(); // Syncing queue info
        //update contri --> curr_pr
        forAll(hd_prdata().queue2, //hd_prdata().num_active
                UpdateContriCopy { hd_prdata });
        hd_prdata.sync(); // Syncing queue info
        //move dlt
        forAll(hd_prdata().queueDlt, //hd_prdata().num_active
                UpdateDltCopy { hd_prdata });
        hd_prdata.sync(); // Syncing queue info
        if(i == 0){
        //move dlt
            forAll(hd_prdata().queue1, //hd_prdata().num_active
                    RemoveContri { hd_prdata });
        }
        hd_prdata.sync(); // Syncing queue info       
        i++;
    }

    //TO DO: copy curr_pr to prev_pr
    //???????????????????????????????????????????????
    //hostPRData.queue2.setQueueCurr(0); 
    //prevEnd = hostPRData.queue2.getQueueEnd();
    forAll(hd_prdata().queue2, //hd_prdata().num_active
            PrevEqCurr { hd_prdata });

//(recompute)--------------------
    //TO DO:: condition for while loop -- h_out is static pr member..
    //while(hd_prdata().iteration < hd_prdata().iterationMax)// && h_out>hostPRData.threshold)
    {
        if (hd_prdata().queue2.size() != 0) {
        forAllVertices(hornet, hd_prdata().queue2, //hd_prdata().num_active
                ComputeContribuitionPerVertex { hd_prdata });
        forAllEdges(hornet, hd_prdata().queue2, //hd_prdata().num_active
                AddContribuitionsUndirected { hd_prdata }, load_balancing);
        forAll(hd_prdata().queue2, //hd_prdata().num_active
                DampAndDiffAndCopy { hd_prdata });
        forAllVertices(hornet, hd_prdata().queue2, //hd_prdata().num_active
                ComputeContribuitionPerVertex { hd_prdata });
        forAllnumV(hornet, Sum { hd_prdata });

        hd_prdata.sync(); // Syncing queue info   
        }

        //host::copyFromDevice(hd_prdata().reduction_out, h_out);
        //hd_prdata().iteration++;
    }


#if 0    
    while (hd_prdata().iteration < hd_prdata().max_iteration &&
           hd_prdata().iteration < hd_prdata().iteration_static) {
        hd_prdata().alphaI = std::pow(hd_prdata().alpha,
                                        hd_prdata().iteration);

        forAll(hd_prdata().active_queue, //hd_prdata().num_active
               InitActiveNewPaths { hd_prdata });

        // Undirected graphs and directed graphs need to be dealt with differently.
        if (!is_directed) {
            forAllEdges(hornet, hd_prdata().active_queue,
                        FindNextActive { hd_prdata }, load_balancing);
            hd_prdata.sync(); // Syncing queue info

            forAllEdges(hornet, hd_prdata().active_queue,
                        UpdateActiveNewPaths { hd_prdata }, load_balancing);
        }
        else {
            forAllEdges(inverted_graph, hd_prdata().active_queue,
                        FindNextActive { hd_prdata }, load_balancing);
            hd_prdata.sync();
            
            forAllEdges(inverted_graph, hd_prdata().active_queue,
                        UpdateActiveNewPaths { hd_prdata }, load_balancing);
        }
        hd_prdata.sync(); // Syncing queue info

        // Checking if we are dealing with a batch of insertions or deletions.
        if (is_insert) {
            forAllEdges(hornet, batch_update,
                        UpdateNewPathsBatchInsert { hd_prdata });
        }
        else {
            forAllEdges(hornet, batch_update,
                        UpdateNewPathsBatchDelete { hd_prdata });
        }
        hd_prdata.sync();

        forAll(hd_prdata().active_queue, UpdatePrevWithCurr { hd_prdata });
        hd_prdata.sync();

        hd_prdata().iteration++;
    }

    if (hd_prdata().iteration > 2) {
        forAll(hd_prdata().active_queue, //hd_prdata().num_active?
               UpdateLastIteration { hd_prdata } );
        hd_prdata.sync();
    }
    // Resetting the fields of the dynamic graph algorithm for all the vertices
    // that were active
    //hd_prdata().num_active ??
    forAll(hd_prdata().active_queue,  InitStreaming {hd_prdata});
#endif



}

//------------------------------------------------------------------------------
int PageRankDynamic::get_iteration_count(){
    //printf("22222\n");
    return hd_prdata().iteration;
}

void PageRankDynamic::batchUpdateInserted(BatchUpdate &batch_update) {
    processUpdate(batch_update, true);
    //printf("33333\n");
}

void PageRankDynamic::batchUpdateDeleted(BatchUpdate &batch_update) {
    processUpdate(batch_update, false);
    //printf("44444\n");
}

void PageRankDynamic::copyPRToHost(double* host_array) {
    //printf("55555\n");
//    pr_static.copyKCToHost(host_array);
}

void PageRankDynamic::copyNumPathsToHost(ulong_t* host_array) {
    //printf("66666\n");
//    pr_static.copyNumPathsToHost(host_array);
}

}// hornets_nest namespace

