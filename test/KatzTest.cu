#include "Static/KatzCentrality/Katz.cuh"
#include "Support/Device/Timer.cuh"


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
 
    // length_t nv, ne,*off;
    // vertexId_t *adj;

	int maxIterations=20;
	int topK=100;



    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT);

    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_ptr(),
                                 graph.out_edges_ptr());

    // auto weights = new int[graph.nE()]();
    // custinger_init.insertEdgeData(weights);
	cuStinger custiger_graph(custinger_init);

	// Finding largest vertex
	vertexId_t maxV=custiger_graph.max_out_degree_id();
	length_t   maxDeg=custiger_graph.max_out_degree();
dasdssasasd
asdasd

    // runKtruss(custinger_init, alg, maxk, graph.name());

	float totalTime;

	// katzCentrality kcPostUpdate;	
	// kcPostUpdate.setInitParameters(maxIterations,topK,maxLen,false);
	// kcPostUpdate.Init(custing);
	// kcPostUpdate.Reset();
	// start_clock(ce_start, ce_stop);
	// kcPostUpdate.Run(custing);
	// totalTime = end_clock(ce_start, ce_stop);
	// cout << "The number of iterations      : " << kcPostUpdate.getIterationCount() << endl;
	// cout << "Total time for KC             : " << totalTime << endl; 
	// cout << "Average time per iteartion    : " << totalTime/(float)kcPostUpdate.getIterationCount() << endl; 

	// kcPostUpdate.Release();
	custing.freecuStinger();

	cudaDeviceReset();

    // delete[] weights;

    return 0;	

}


	

}

