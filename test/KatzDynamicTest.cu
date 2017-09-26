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

#include "Static/KatzCentrality/katz.cuh"
#include "Dynamic/KatzCentrality/katz.cuh"
#include "Device/Timer.cuh"


using namespace timer;
using namespace custinger;
using namespace custinger_alg;

int main(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

	int device=0;

    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);


	int maxIterations=1000;
	int topK=100;

    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT);

	cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_ptr(),
                                 graph.out_edges_ptr());

	cuStinger custinger_graph(custinger_init);


	// Finding largest vertex
	degree_t   maxDeg		=custinger_graph.max_degree();
	cout << "Max degree is " << maxDeg << endl;

	float totalTime;

	custinger_alg::katzCentrality kcPostUpdate(custinger_graph);
	kcPostUpdate.setInitParameters(maxIterations,topK,maxDeg,true);
	kcPostUpdate.init();
	kcPostUpdate.reset();
	Timer<DEVICE> TM;
	TM.start();
	kcPostUpdate.run();
	TM.stop();
	totalTime = TM.duration();
	cout << "The number of iterations      : " << kcPostUpdate.getIterationCount() << endl;
	cout << "Total time for KC             : " << totalTime << endl;
	cout << "Average time per iteartion    : " << totalTime/(float)kcPostUpdate.getIterationCount() << endl;

	kcPostUpdate.release();

    return 0;

}
