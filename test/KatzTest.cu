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
#include <Device/Timer.cuh>
#include <GraphIO/GraphStd.hpp>

int main(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;
    using namespace hornets_nest;
    using namespace timer;

    int max_iterations = 1000;
    int           topK = 100;

    GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT_INFO);

    HornetInit hornet_init(graph.nV(), graph.nE(),
                           graph.out_offsets_ptr(),
                           graph.out_edges_ptr());

    HornetGraph hornet_graph(hornet_init);

    // Finding largest vertex
    degree_t max_degree_vertex = hornet_graph.max_degree_id();
    std::cout << "Max degree vextex is " << max_degree_vertex << std::endl;

    KatzCentrality kcPostUpdate(hornet_graph, max_iterations, topK,
                                max_degree_vertex);

    Timer<DEVICE> TM;
    TM.start();

    kcPostUpdate.run();

    TM.stop();

    auto total_time = TM.duration();
    std::cout << "The number of iterations   : "
              << kcPostUpdate.get_iteration_count()
              << "\nTotal time for KC          : " << total_time
              << "\nAverage time per iteartion : "
              << total_time /
                 static_cast<float>(kcPostUpdate.get_iteration_count())
              << "\n";
}
