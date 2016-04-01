 


#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

 

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.hpp"
// #include "update.hpp"
// #include "cuStinger.hpp"


using namespace std;

void readGraphDIMACS(char* filePath, int32_t** prmoff, int32_t** prmind, int32_t* prmnv, int32_t* prmne);


//Note: Times are returned in seconds
void start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&end));
	checkCudaErrors(cudaEventRecord(start,0));
}

float end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	checkCudaErrors(cudaEventRecord(end,0));
	checkCudaErrors(cudaEventSynchronize(end));
	checkCudaErrors(cudaEventElapsedTime(&time,start,end));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(end));

	return time/(float)1000;
}


void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
	for(int32_t e=0; e<numEdges; e++){
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;
	}
}


int main(const int argc, char *argv[])
{
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	cout << "Name : " << prop.name <<  endl;
	// cout << "computeMode : " << prop.computeMode <<  endl;
 // 	cout << "gridsize.x : " << prop.maxGridSize[0] <<  endl;
 // 	cout << "gridsize.y : " << prop.maxGridSize[1] <<  endl;
 // 	cout << "gridsize.z : " << prop.maxGridSize[2] <<  endl;
 
    length_t nv, ne,*off;
    vertexId_t *adj;

    cout << argv[1] << endl;

	int numEdges=10000;
	if(argc>2)
		numEdges=atoi(argv[2]);
	srand(100);

    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne);
// 
	cout << "Vertices " << nv << endl;
	cout << "Edges " << ne << endl;

	// int32t *d_utilized,*d_max,**d_adj;

	cudaEvent_t ce_start,ce_stop;

	// cuStinger custing(stingyInitAllocater,stingyUpdateAllocater);

	// cuStinger custing2;

	// cuStingerInitConfig cuInit;
	// cuInit.initState =eInitStateCSR;
	// cuInit.maxNV = nv+1;
	// cuInit.useVWeight = false;
	// cuInit.isSemantic = false;  // Use edge types and vertex types
	// cuInit.useEWeight = false;

	// // CSR data
	// cuInit.csrNV 			= nv;
	// cuInit.csrNe   			= ne;
	// cuInit.csrOff 			= off;
	// cuInit.csrAdj 			= adj;
	// cuInit.csrVW 			= NULL;
	// cuInit.csrEW			= NULL;

	// start_clock(ce_start, ce_stop);
	// custing2.initializeCuStinger(cuInit);
	// cout << "Allocation and Copy Time : " << end_clock(ce_start, ce_stop) << endl;



	cuStinger custing;

	start_clock(ce_start, ce_stop);
	custing.initializeCuStinger(nv,ne,off,adj);
	cout << "Allocation and Copy Time : " << end_clock(ce_start, ce_stop) << endl;

	cout << "Host utilized   : " << custing.getNumberEdgesUsed() << endl;

	length_t numEdgesL = numEdges;
	BatchUpdateData bud(numEdgesL,true);

	generateEdgeUpdates(nv, numEdges, bud.getSrc(),bud.getDst());
	BatchUpdate bu(bud);

	start_clock(ce_start, ce_stop);
		// update(custing,bu);
	cout << "Update time     : " << end_clock(ce_start, ce_stop) << endl;

	cout << "Host utilized   : " << custing.getNumberEdgesUsed() << endl;

	custing.freecuStinger();
	
	// custing2.freecuStinger();

    return 0;	
}       

