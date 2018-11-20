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
#include "Dynamic/KatzCentrality/Katz.cuh"
#include "KatzOperators.cuh"

namespace hornets_nest {

KatzCentralityDynamic::KatzCentralityDynamic(HornetGraph& hornet,
                                             HornetGraph& inverted_graph,
                                             int max_iteration, int K,
                                             degree_t max_degree) :
                                   StaticAlgorithm(hornet),
                                   load_balancing(hornet),
                                   inverted_graph(inverted_graph),
                                   is_directed(false),
                                   kc_static(hornet, max_iteration, K,
                                             max_degree, false) {

    hd_katzdata().active_queue.initialize(hornet);

    gpu::allocate(hd_katzdata().new_paths_curr, hornet.nV());
    gpu::allocate(hd_katzdata().new_paths_prev, hornet.nV());
    gpu::allocate(hd_katzdata().active,         hornet.nV());

    hd_katzdata = kc_static.katz_data();

    std::cout << "Oded remember to take care of memory de-allocation\n"
              << "Oded need to figure out correct API for dynamic graph"
              << "algorithms\n"
              << "Dynamic katz centrality algorithm needs to get both the"
              << "original graph and the inverted graph for directed graphs"
              << std::endl;
}

KatzCentralityDynamic::KatzCentralityDynamic(HornetGraph& hornet,
                                             int max_iteration, int K,
                                             degree_t max_degree) :
                                   StaticAlgorithm(hornet),
                                   load_balancing(hornet),
                                   inverted_graph(inverted_graph),
                                   is_directed(true),
                                   kc_static(inverted_graph, max_iteration, K,
                                             max_degree, true) {

    hd_katzdata().active_queue.initialize(hornet);

    gpu::allocate(hd_katzdata().new_paths_curr, hornet.nV());
    gpu::allocate(hd_katzdata().new_paths_prev, hornet.nV());
    gpu::allocate(hd_katzdata().active,         hornet.nV());

    hd_katzdata = kc_static.katz_data();

    std::cout << "Oded remember to take care of memory de-allocation\n"
              << "Oded need to figure out correct API for dynamic graph"
              << "algorithms\n"
              << "Dynamic katz centrality algorithm needs to get both the"
              << "original graph and the inverted graph for directed graphs"
              << std::endl;
}

KatzCentralityDynamic::~KatzCentralityDynamic() {
    release();
}

void KatzCentralityDynamic::run_static() {
    // Executing the static graph algorithm
    kc_static.reset();
    kc_static.run();

    hd_katzdata().iteration_static = hd_katzdata().iteration;

    // Initializing the fields of the dynamic graph algorithm
    forAllnumV(hornet, InitStreaming { hd_katzdata } );
}

void KatzCentralityDynamic::release(){
    gpu::free(hd_katzdata().new_paths_curr);
    gpu::free(hd_katzdata().new_paths_prev);
    gpu::free(hd_katzdata().active);
}

//==============================================================================

void KatzCentralityDynamic::processUpdate(BatchUpdate& batch_update,
                                          bool is_insert) {
    // Resetting the queue of the active vertices.
    hd_katzdata().active_queue.clear();
    hd_katzdata().iteration = 1;

    // Initialization of insertions or deletions is slightly different.
    if (is_insert)
        forAllEdges(hornet, batch_update, SetupInsertions { hd_katzdata });
    else
        forAllEdges(hornet, batch_update, SetupDeletions { hd_katzdata } );
    hd_katzdata.sync();

    hd_katzdata().iteration = 2;

    while (hd_katzdata().iteration < hd_katzdata().max_iteration &&
           hd_katzdata().iteration < hd_katzdata().iteration_static) {
        hd_katzdata().alphaI = std::pow(hd_katzdata().alpha,
                                        hd_katzdata().iteration);

        forAll(hd_katzdata().active_queue, //hd_katzdata().num_active
               InitActiveNewPaths { hd_katzdata });

        // Undirected graphs and directed graphs need to be dealt with differently.
        if (!is_directed) {
            forAllEdges(hornet, hd_katzdata().active_queue,
                        FindNextActive { hd_katzdata }, load_balancing);
            hd_katzdata.sync(); // Syncing queue info

            forAllEdges(hornet, hd_katzdata().active_queue,
                        UpdateActiveNewPaths { hd_katzdata },
                        load_balancing );
        }
        else {
            forAllEdges(inverted_graph, hd_katzdata().active_queue,
                        FindNextActive { hd_katzdata }, load_balancing);
            hd_katzdata.sync();
            forAllEdges(inverted_graph, hd_katzdata().active_queue,
                        UpdateActiveNewPaths { hd_katzdata }, load_balancing);
        }
        hd_katzdata.sync(); // Syncing queue info

        // Checking if we are dealing with a batch of insertions or deletions.
        if (is_insert) {
            forAllEdges(hornet, batch_update,
                        UpdateNewPathsBatchInsert { hd_katzdata });
        }
        else {
            forAllEdges(hornet, batch_update,
                        UpdateNewPathsBatchDelete { hd_katzdata });
        }
        hd_katzdata.sync();

        forAll(hd_katzdata().active_queue, UpdatePrevWithCurr { hd_katzdata });
        hd_katzdata.sync();

        hd_katzdata().iteration++;
    }

    if (hd_katzdata().iteration > 2) {
        forAll(hd_katzdata().active_queue, //hd_katzdata().num_active?
               UpdateLastIteration { hd_katzdata } );
        hd_katzdata.sync();
    }
    // Resetting the fields of the dynamic graph algorithm for all the vertices
    // that were active
    //hd_katzdata().num_active ??
    forAll(hd_katzdata().active_queue,  InitStreaming {hd_katzdata});
}

//------------------------------------------------------------------------------
int KatzCentralityDynamic::get_iteration_count(){
    return hd_katzdata().iteration;
}

void KatzCentralityDynamic::batchUpdateInserted(BatchUpdate &batch_update) {
    processUpdate(batch_update, true);
}

void KatzCentralityDynamic::batchUpdateDeleted(BatchUpdate &batch_update) {
    processUpdate(batch_update, false);
}

void KatzCentralityDynamic::copyKCToHost(double* host_array) {
    kc_static.copyKCToHost(host_array);
}

void KatzCentralityDynamic::copyNumPathsToHost(ulong_t* host_array) {
    kc_static.copyNumPathsToHost(host_array);
}

}// cuStingerAlgs namespace
