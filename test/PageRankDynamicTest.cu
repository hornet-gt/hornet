/**
 * @internal
 * @author Euna Kim                                                  <br>
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
#include "Dynamic/PageRank/PageRank.cuh"
#include <Graph/GraphStd.hpp>
#include <Device/Util/Timer.cuh>
#include <Util/CommandLineParam.hpp>

//added
#include "Hornet.hpp"
#include "Core/GPUHornet/BatchUpdate.cuh"
#include "Util/BatchFunctions.hpp"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>


int main(int argc, char* argv[]) {

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;

    using namespace timer;
    using namespace hornets_nest;

//    graph::GraphStd<vid_t, eoff_t> graph;
    GraphStd<vid_t, eoff_t> graph(UNDIRECTED | ENABLE_INGOING);
    graph.read(argv[1], SORT | PRINT_INFO);


//    CommandLineParam cmd(graph, argc, argv);

    std::cerr<<"Init hornet\n";
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());
    std::cerr<<"Init hornet gpu\n";
    HornetGPU hornet_graph(hornet_init);

    std::cerr<<"Init PageRankDynamic\n";
    PageRankDynamic page_rank(hornet_graph, 5, 0.001,0.85);
//    StaticPageRank page_rank(hornet_graph, 5, 0.001, 0.85);


    std::cerr<<"Page Rank Run\n";
    Timer<DEVICE> TM;
    TM.start();
//    page_rank.run();
    //page_rank.run();
    TM.stop();
    TM.print("PageRank Static");


//batch <start>---------------------------------------------------------------
    int batch_size = 20;//std::stoi(argv[2]);
    vid_t* batch_src, *batch_dst;
    cuMallocHost(batch_src, batch_size);
    cuMallocHost(batch_dst, batch_size);
    generateBatch(graph, batch_size, batch_src, batch_dst,
                      BatchGenType::INSERT, batch_gen_property::UNIQUE);
    //vid_t batch_src[] = { 0, 0, 2 };
    //vid_t batch_dst[] = { 2, 6, 7 };
    gpu::BatchUpdate batch_update(batch_src, batch_dst, batch_size);

    batch_update.print();
    std::cout << "------------------------------------------------" <<std::endl;

    using namespace gpu::batch_property;
    //hornet_graph.reserveBatchOpResource(batch_size);
    hornet_graph.insertEdgeBatch(batch_update);
    hornet_graph.deleteEdgeBatch(batch_update);
    cuFreeHost(batch_src);
    cuFreeHost(batch_dst);
//batch <end>---------------------------------------------------------------

    TM.start();
    page_rank.processUpdate(batch_update, true);  //batch, isinsertion
    TM.stop();
    TM.print("PageRank Update");

    //auto is_correct = page_rank.validate();
    auto is_correct = true;
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");

    return 0;
}

