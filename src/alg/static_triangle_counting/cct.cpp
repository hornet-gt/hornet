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

void callDeviceAllTrianglesCSR(const vertexId_t nv,
    length_t const * const __restrict__ d_off, vertexId_t const * const __restrict__ d_ind,
    int * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);


// CPU Version - assume sorted index lists. 
int hostSingleIntersection (const vertexId_t ai, const length_t alen, const vertexId_t * a,
						    const vertexId_t bi, const length_t blen, const vertexId_t * b){
	length_t ka = 0, kb = 0,out = 0;
	if (!alen || !blen || a[alen-1] < b[0] || b[blen-1] < a[0])
    	return 0;

	while (1) {
    	if (ka >= alen || kb >= blen) break;
		vertexId_t va = a[ka],vb = b[kb];

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

void hostCountTriangles (const vertexId_t nv, const length_t * off,
    const vertexId_t * ind, int * triNE, int64_t* allTriangles)
{
	int32_t edge=0;
	int64_t sum=0;
    for (vertexId_t src = 0; src < nv; src++)
    {
		length_t srcLen=off[src+1]-off[src];
		for(int iter=off[src]; iter<off[src+1]; iter++)
		{
			vertexId_t dest=ind[iter];
			length_t destLen=off[dest+1]-off[dest];			
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
	return sum;
}

int comparecuStingerAndCSR(cuStinger& custing, vertexId_t nv,length_t ne, length_t*  off,vertexId_t*  ind)
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
    length_t *d_off = NULL;
    vertexId_t* d_ind = NULL;
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
		CUDA(cudaMalloc(&d_off, sizeof(length_t)*(nv+1)));
		CUDA(cudaMalloc(&d_ind, sizeof(vertexId_t)*ne));
		CUDA(cudaMalloc(&d_triangles, sizeof(triangle_t)*(nv+1)));

		CUDA(cudaMemcpy(d_off, off, sizeof(length_t)*(nv+1), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(d_ind, ind, sizeof(vertexId_t)*ne, cudaMemcpyHostToDevice));

		triangle_t* h_triangles = (triangle_t *) malloc ( sizeof(triangle_t)*(nv+1)  );		

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

					CUDA(cudaMemcpy(d_triangles, h_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));
					start_clock(ce_start, ce_stop);
						callDeviceAllTrianglesCSR(nv,d_off, d_ind, d_triangles, tsp,nbl,shifter,blocks, sps);
					time = end_clock(ce_start, ce_stop);
					CUDA(cudaMemcpy(h_triangles, d_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));

					if(time<minTime) minTime=time; 
					sumDevice=sumTriangleArray(h_triangles,nv);initHostTriangleArray(h_triangles,nv);
					printf("!!! %d %d %d %d %d \t\t %ld \t %f\n", blocks,sps, tsp, nbl, shifter,sumDevice, time);

					CUDA(cudaMemcpy(d_triangles, h_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));
					start_clock(ce_start, ce_stop);
						callDeviceAllTriangles(custing, d_triangles, tsp,nbl,shifter,blocks, sps);
					time = end_clock(ce_start, ce_stop);
					CUDA(cudaMemcpy(h_triangles, d_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));

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
