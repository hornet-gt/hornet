
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>



#include "Static/TriangleCounting/triangle.cuh"
#include "Support/Device/Timer.cuh"


using namespace std;
using namespace timer;
using namespace custinger;
using namespace custinger_alg;

#define STAND_PRINTF(sys, time, triangles) printf("%s : \t%ld \t%f\n", sys,triangles, time);

namespace custinger_alg { 

int arrayBlocks[]={16000};
int arrayBlockSize[]={32,64,96,128,192,256};
int arrayThreadPerIntersection[]={1,2,4,8,16,32};
int arrayThreadShift[]={0,1,2,3,4,5};
// int arrayBlocks[]={64000};
// int arrayBlockSize[]={256};
// int arrayThreadPerIntersection[]={32};
// int arrayThreadShift[]={5};


void initHostTriangleArray(triangle_t* h_triangles, vid_t nv){	
	for(vid_t sd=0; sd<(nv);sd++){
		h_triangles[sd]=0;
	}
}

int64_t sumTriangleArray(triangle_t* h_triangles, vid_t nv){	
	int64_t sum=0;
	for(vid_t sd=0; sd<(nv);sd++){
	  sum+=h_triangles[sd];
	}
	return sum;
}

int comparecuStingerAndCSR(cuStinger& custing, vid_t nv,degree_t ne)
{
	int device = 0;
	int run    = 2;
		
	// triangle_t *d_triangles = NULL;  

	triangle_t* h_triangles = (triangle_t *) malloc ( sizeof(triangle_t)*(nv+1)  );		

	float minTime=10e9,time,minTimecuStinger=10e9;

	int blocksToTest=sizeof(arrayBlocks)/sizeof(int);
	int blockSizeToTest=sizeof(arrayBlockSize)/sizeof(int);
	int tSPToTest=sizeof(arrayThreadPerIntersection)/sizeof(int);
	for(int b=0;b<blocksToTest; b++){
		int blocks=arrayBlocks[b];
		for(int bs=0; bs<blockSizeToTest; bs++){
			int sps=arrayBlockSize[bs];
			for(int t=0; t<tSPToTest;t++){
				int tsp=arrayThreadPerIntersection[t];
				int shifter=arrayThreadShift[t];
				int nbl=sps/tsp;

				Timer<DEVICE> TM;

				// cudaMemcpy(d_triangles, h_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice);
				TM.start();
					// callDeviceAllTriangles(custing, d_triangles, tsp,nbl,shifter,blocks, sps);
				TM.stop();
				time = TM.duration();
				// cudaMemcpy(h_triangles, d_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost);

				triangle_t sumDevice=0;
				if(time<minTimecuStinger) minTimecuStinger=time; 
				// sumDevice=sumTriangleArray(h_triangles,nv);initHostTriangleArray(h_triangles,nv);

				printf("### %d %d %d %d %d \t\t %ld \t %f\n", blocks,sps, tsp, nbl, shifter,sumDevice, time);
			}
		}	
	}
	cout << nv << ", " << ne << ", "<< minTime << ", " << minTimecuStinger<< endl;
}

}// cuStingerAlgs namespace


int main(const int argc, char *argv[]){
 
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT);

	cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_ptr(),
                                 graph.out_edges_ptr());

	cuStinger custiger_graph(custinger_init);

	// comparecuStingerAndCSR(custing,nv,ne,off,adj);

    return 0;	
}

