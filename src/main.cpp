 


#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.h"
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


void generateEdgeUpdates(int32_t nv, int32_t numEdges, int32_t* edgeSrc, int32_t* edgeDst){

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
 	// printf("  Device name: %s\n", prop.name);
	cout << "Name : " << prop.name <<  endl;
	// cout << "computeMode : " << prop.computeMode <<  endl;
 // 	cout << "gridsize.x : " << prop.maxGridSize[0] <<  endl;
 // 	cout << "gridsize.y : " << prop.maxGridSize[1] <<  endl;
 // 	cout << "gridsize.z : " << prop.maxGridSize[2] <<  endl;
 
 // 	cout << "Major      : " << prop.major <<  endl;
 // 	cout << "Minor      : " << prop.minor <<  endl;

    int32_t nv, ne,*off,*adj;

    cout << argv[1] << endl;

	int numEdges=10000;
	if(argc>2)
		numEdges=atoi(argv[2]);
	srand(100);

    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne);

	cout << "Vertices " << nv << endl;
	cout << "Edges " << ne << endl;

	int32_t *d_adjSizeUsed,*d_adjSizeMax,**d_adjArray;

	cudaEvent_t ce_start,ce_stop;

	cuStinger custing;
	start_clock(ce_start, ce_stop);
	custing.hostCsrTocuStinger(nv,ne,off,adj);
	cout << "Allocation and Copy Time : " << end_clock(ce_start, ce_stop) << endl;

	BatchUpdate bu(numEdges);
	generateEdgeUpdates(nv, numEdges, bu.getHostSrcArray(),bu.getHostDstArray());
	bu.resetHostIndCount();
	bu.copyHostToDevice();

	start_clock(ce_start, ce_stop);
		update(custing,bu);
	cout << "Update time     : " << end_clock(ce_start, ce_stop) << endl;

	bu.copyDeviceToHost();

	cout << "Number of unsuccessful insertions : " << bu.getHostIndCount() << endl;

	int32_t sum=0, *tempsrc=bu.getHostSrcArray(),*tempdst=bu.getHostDstArray();

	for(int i=0; i<numEdges; i++){
		sum+=tempdst[i]+tempsrc[i];
		if(sum!=0){
			printf("First broken edge : %d\n",i);
			break;
		}
	}

	cout << "Checking that everything is zero : " << sum << endl;

	// int32_t** h_adjArray = (int32_t**)allocHostArray(nv, sizeof(int32_t*));
	// copyArrayDeviceToHost(d_adjArray,h_adjArray,nv, sizeof(int32_t*));
	// for(int v = 0; v < nv; v++){
 //        freeDeviceArray(h_adjArray[v]); 
 //    }

	// freeDeviceArray(d_adjArray);
	// freeDeviceArray(d_adjSizeUsed);
	// freeDeviceArray(d_adjSizeMax);

	custing.freecuStinger();


    return 0;	
}       


