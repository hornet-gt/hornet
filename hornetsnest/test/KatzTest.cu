/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
 *   ogreen@gatech.edu
 * @date August, 2017
 * @version v2
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
 * @file
 */
#include "Static/KatzCentrality/Katz.cuh"
#include <StandardAPI.hpp>
#include <Device/Util/Timer.cuh>
#include <Graph/GraphStd.hpp>

int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;
    using namespace hornets_nest;
    using namespace timer;

	// Limit the number of iteartions for graphs with large number of vertices.
    int max_iterations = 50;


	cudaSetDevice(0);
    GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    
    graph.read(argv[1], SORT | PRINT_INFO);

    HornetInit hornet_init(graph.nV(), graph.nE(),
                           graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
	// Users can add the number of TopK vertices for the approximation
	int           topK = graph.nV();
     if(argc>2)
        topK=atoi(argv[2]);
 
    // Finding largest vertex degree
    degree_t max_degree_vertex = hornet_graph.max_degree();
    std::cout << "Max degree vextex is " << max_degree_vertex << std::endl;


    KatzCentrality kcPostUpdate(hornet_graph, max_iterations, topK,
                                max_degree_vertex);

    Timer<DEVICE> TM;
    TM.start();

    kcPostUpdate.run();

    TM.stop();

    auto total_time = TM.duration();
    std::cout << "The number of iterations     : "
              << kcPostUpdate.get_iteration_count()
              << "\nTopK                       : " << topK 
              << "\nTotal time for KC          : " << total_time
              << "\nAverage time per iteartion : "
              << total_time /
                 static_cast<float>(kcPostUpdate.get_iteration_count())
              << "\n";

    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
#if defined(RMM_WRAPPER)
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

