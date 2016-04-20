#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "cct.hpp"


#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)

void callDeviceAllTrianglesCSR(const int nv,
    int const * const __restrict__ d_off, int const * const __restrict__ d_ind,
    int * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);


// CPU Version - assume sorted index lists. 
int hostSingleIntersection (const int32_t ai, const int32_t alen, const int32_t * a,
						    const int32_t bi, const int32_t blen, const int32_t * b){
	int32_t ka = 0, kb = 0,out = 0;
	if (!alen || !blen || a[alen-1] < b[0] || b[blen-1] < a[0])
    	return 0;

	while (1) {
    	if (ka >= alen || kb >= blen) break;
		int32_t va = a[ka],vb = b[kb];

	    // If you now that you don't have self edges then you don't need to check for them and you can get better performance.
		#if(0)
		    // Skip self-edges.
		    if ((va == ai)) {
		      ++ka;
		      if (ka >= alen) break;
		      va = a[ka];
		    }
		    if ((vb == bi)) {
		      ++kb;
		      if (kb >= blen) break;
		      vb = b[kb];
		    }
		#endif

	    if (va == vb) {
	     	++ka; ++kb; ++out;
	    }
	    else if (va < vb) {
	      ++ka;
	      while (ka < alen && a[ka] < vb) ++ka;
	    } else {
	      ++kb;
	      while (kb < blen && va > b[kb]) ++kb;
	    }
	}
	return out;
}

void hostCountTriangles (const int32_t nv, const int32_t * off,
    const int32_t * ind, int * triNE, int64_t* allTriangles)
{
	int32_t edge=0;
	int64_t sum=0;
    for (int src = 0; src < nv; src++)
    {
		int srcLen=off[src+1]-off[src];
		for(int iter=off[src]; iter<off[src+1]; iter++)
		{
			int dest=ind[iter];
			int destLen=off[dest+1]-off[dest];			
			triNE[edge]= hostSingleIntersection (src, srcLen, ind+off[src],
													dest, destLen, ind+off[dest]);
			sum+=triNE[edge++];
		}
	}	
	*allTriangles=sum;
}

#define STAND_PRINTF(sys, time, triangles) printf("%s : \t%ld \t%f\n", sys,triangles, time);

#define PAR_FILENAME 1
#define PAR_DEVICE   2
#define PAR_RUN      3
#define PAR_BLOCKS   4
#define PAR_SP       5
#define PAR_T_SP     6
#define PAR_NUM_BL   7
#define PAR_SHIFT    8


int arrayBlocks[]={16000};
int arrayBlockSize[]={32,64,96,128,192,256};
int arrayThreadPerIntersection[]={1,2,4,8,16,32};
int arrayThreadShift[]={0,1,2,3,4,5};


void initHostTriangleArray(triangle_t* h_triangles, vertexId_t nv){	
	for(vertexId_t sd=0; sd<(nv);sd++){
		h_triangles[sd]=0;
	}
}

int64_t sumTriangleArray(triangle_t* h_triangles, vertexId_t nv){	
	int64_t sum=0;
	for(vertexId_t sd=0; sd<(nv);sd++){
	  sum+=h_triangles[sd];
	}
}

int comparecuStingerAndCSR(cuStinger& custing, int nv,int ne, int32_t*  off,int32_t*  ind)
{
	int device = 0;
	int run    = 3; 
//  int scriptMode =atoi(argv[PAR_SCRIPT]);
//	int sps =atoi(argv[PAR_SP]);	
//	int tsp =atoi(argv[PAR_T_SP]);	
//	int nbl =atoi(argv[PAR_NUM_BL]);
//	int shifter =atoi(argv[PAR_SHIFT]);
		
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties	(&prop,device);	
    int *d_off = NULL,*d_ind = NULL;
	triangle_t *d_triangles = NULL;  

   	int * triNE = (int *) malloc ((ne ) * sizeof (int));	
	int64_t allTrianglesCPU=0;
	
	if(run&1){
		cudaEvent_t startCPU, stopCPU;
		float timeCPU;
		cudaEventCreate(&startCPU); cudaEventCreate(&stopCPU);
		cudaEventRecord(startCPU, 0);
		hostCountTriangles (nv, off,ind, triNE, &allTrianglesCPU);
		cudaEventRecord(stopCPU, 0);cudaEventSynchronize(stopCPU);
		
		cudaThreadSynchronize();cudaEventElapsedTime(&timeCPU, startCPU, stopCPU);
		STAND_PRINTF("CPU", timeCPU,allTrianglesCPU)
	}	

	if(run&2){
		cudaSetDevice(device);
		CUDA(cudaMalloc(&d_off, sizeof(int)*(nv+1)));
		CUDA(cudaMalloc(&d_ind, sizeof(int)*ne));
		CUDA(cudaMalloc(&d_triangles, sizeof(triangle_t)*(nv+1)));

		CUDA(cudaMemcpy(d_off, off, sizeof(int)*(nv+1), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(d_ind, ind, sizeof(int)*ne, cudaMemcpyHostToDevice));

		int* h_triangles = (int *) malloc ( sizeof(int)*(nv+1)  );		

		float minTime=10e9,time,minTimecuStinger=10e9;

		int64_t sumDevice=0;
		initHostTriangleArray(h_triangles,nv);

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

					cudaEvent_t ce_start,ce_stop;

					CUDA(cudaMemcpy(d_triangles, h_triangles, sizeof(int)*(nv+1), cudaMemcpyHostToDevice));
					start_clock(ce_start, ce_stop);
						callDeviceAllTrianglesCSR(nv,d_off, d_ind, d_triangles, tsp,nbl,shifter,blocks, sps);
					time = end_clock(ce_start, ce_stop);
					CUDA(cudaMemcpy(h_triangles, d_triangles, sizeof(int)*(nv+1), cudaMemcpyDeviceToHost));

					if(time<minTime) minTime=time; 
					sumDevice=sumTriangleArray(h_triangles,nv);initHostTriangleArray(h_triangles,nv);
					printf("!!! %d %d %d %d %d \t\t %ld \t %f\n", blocks,sps, tsp, nbl, shifter,sumDevice, time);

					CUDA(cudaMemcpy(d_triangles, h_triangles, sizeof(int)*(nv+1), cudaMemcpyHostToDevice));
					start_clock(ce_start, ce_stop);
						callDeviceAllTriangles(custing, d_triangles, tsp,nbl,shifter,blocks, sps);
					time = end_clock(ce_start, ce_stop);
					CUDA(cudaMemcpy(h_triangles, d_triangles, sizeof(int)*(nv+1), cudaMemcpyDeviceToHost));

					if(time<minTimecuStinger) minTimecuStinger=time; 
					sumDevice=sumTriangleArray(h_triangles,nv);initHostTriangleArray(h_triangles,nv);

					printf("### %d %d %d %d %d \t\t %ld \t %f\n", blocks,sps, tsp, nbl, shifter,sumDevice, time);
			    }
			}	
		}
		STAND_PRINTF("GPU - csr     ", minTime,sumDevice)
		STAND_PRINTF("GPU - custing ", minTimecuStinger,sumDevice)

		free(h_triangles);

		CUDA(cudaFree(d_off));
		CUDA(cudaFree(d_ind));
		CUDA(cudaFree(d_triangles));
	}
	free(triNE);
    return 0;
}



int main(const int argc, char *argv[]){
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne,*off;
    vertexId_t *adj;
    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne);
	cout << "Vertices " << nv << endl;
	cout << "Edges " << ne << endl;

	cudaEvent_t ce_start,ce_stop;
	cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

	cuStingerInitConfig cuInit;
	cuInit.initState =eInitStateCSR;
	cuInit.maxNV = nv+1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	// CSR data
	cuInit.csrNV 			= nv;
	cuInit.csrNE	   		= ne;
	cuInit.csrOff 			= off;
	cuInit.csrAdj 			= adj;
	cuInit.csrVW 			= NULL;
	cuInit.csrEW			= NULL;

	custing.initializeCuStinger(cuInit);

	comparecuStingerAndCSR(custing,nv,ne,off,adj);

	custing.freecuStinger();

	free(off);
	free(adj);
    return 0;	
}
