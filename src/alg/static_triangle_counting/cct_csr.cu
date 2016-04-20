

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

	 
__device__ void conditionalWarpReduce(volatile int*sharedData,int blockSize,int dataLength){
  if(blockSize >= dataLength){
    if(threadIdx.x < (dataLength/2))
    {sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
    __syncthreads();
  }
}

__device__ void warpReduce(int* __restrict__ outDataPtr,
    volatile int* __restrict__ sharedData,int blockSize){
  conditionalWarpReduce(sharedData,blockSize,64);
  conditionalWarpReduce(sharedData,blockSize,32);
  conditionalWarpReduce(sharedData,blockSize,16);
  conditionalWarpReduce(sharedData,blockSize,8);
  conditionalWarpReduce(sharedData,blockSize,4);
  if(threadIdx.x == 0)
    {*outDataPtr= sharedData[0] + sharedData[1];}
  __syncthreads();
}

__device__ void conditionalReduce(volatile int* __restrict__ sharedData,int blockSize,int dataLength){
	if(blockSize >= dataLength){
		if(threadIdx.x < (dataLength/2))
		{sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
		__syncthreads();
	}
	if((blockSize < dataLength) && (blockSize > (dataLength/2))){
		if(threadIdx.x+(dataLength/2) < blockSize){
			sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];
		}
		__syncthreads();
	}
}

__device__ void blockReduce(int* __restrict__ outGlobalDataPtr,
    volatile int* __restrict__ sharedData,int blockSize){
  __syncthreads();
  conditionalReduce(sharedData,blockSize,1024);
  conditionalReduce(sharedData,blockSize,512);
  conditionalReduce(sharedData,blockSize,256);
  conditionalReduce(sharedData,blockSize,128);

  warpReduce(outGlobalDataPtr, sharedData, blockSize);
  __syncthreads();
}


__device__ void initialize(const int diag_id, const int u_len, int v_len,
    int* const __restrict__ u_min, int* const __restrict__ u_max,
    int* const __restrict__ v_min, int* const __restrict__ v_max,
    int* const __restrict__ found)
{
	if (diag_id == 0){
		*u_min=*u_max=*v_min=*v_max=0;
		*found=1;
	}
	else if (diag_id < u_len){
		*u_min=0; *u_max=diag_id;
		*v_max=diag_id;*v_min=0;
	}
	else if (diag_id < v_len){
		*u_min=0; *u_max=u_len;
		*v_max=diag_id;*v_min=diag_id-u_len;
	}
	else{
		*u_min=diag_id-v_len; *u_max=u_len;
		*v_min=diag_id-u_len; *v_max=v_len;
	}
}

__device__ void calcWorkPerThread(const int uLength, const int vLength, 
	const int threadsPerIntersection, const int threadId,
    int * const __restrict__ outWorkPerThread, int * const __restrict__ outDiagonalId){
  int totalWork = uLength + vLength;
  int remainderWork = totalWork%threadsPerIntersection;
  int workPerThread = totalWork/threadsPerIntersection;

  int longDiagonals  = (threadId > remainderWork) ? remainderWork:threadId;
  int shortDiagonals = (threadId > remainderWork) ? (threadId - remainderWork):0;

  *outDiagonalId = ((workPerThread+1)*longDiagonals) + (workPerThread*shortDiagonals);
  *outWorkPerThread = workPerThread + (threadId < remainderWork);
}

__device__ void bSearch(unsigned int found, const int diagonalId,
    int const * const __restrict__ uNodes, int const * const __restrict__ vNodes,
    int const * const __restrict__ uLength, 
    int * const __restrict__ outUMin, int * const __restrict__ outUMax,
    int * const __restrict__ outVMin, int * const __restrict__ outVMax,    
    int * const __restrict__ outUCurr,
    int * const __restrict__ outVCurr){
  	int length;
	
	while(!found) {
	    *outUCurr = (*outUMin + *outUMax)>>1;
	    *outVCurr = diagonalId - *outUCurr;
	    if(*outVCurr >= *outVMax){
			length = *outUMax - *outUMin;
			if(length == 1){
				found = 1;
				continue;
			}
	    }

	    unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr-1];
	    unsigned int comp2 = uNodes[*outUCurr-1] > vNodes[*outVCurr];
	    if(comp1 && !comp2){
			found = 1;
	    }
	    else if(comp1){
	      *outVMin = *outVCurr;
	      *outUMax = *outUCurr;
	    }
	    else{
	      *outVMax = *outVCurr;
	      *outUMin = *outUCurr;
	    }
  	}

	if((*outVCurr >= *outVMax) && (length == 1) && (*outVCurr > 0) &&
	(*outUCurr > 0) && (*outUCurr < (*uLength - 1))){
		unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
		unsigned int comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
		if(!comp1 && !comp2){(*outUCurr)++; (*outVCurr)--;}
	}
}

__device__ int fixThreadWorkEdges(const int uLength, const int vLength,
    int * const __restrict__ uCurr, int * const __restrict__ vCurr,
    int const * const __restrict__ uNodes, int const * const __restrict__ vNodes){
	
	unsigned int uBigger = (*uCurr > 0) && (*vCurr < vLength) && (uNodes[*uCurr-1] == vNodes[*vCurr]);
	unsigned int vBigger = (*vCurr > 0) && (*uCurr < uLength) && (vNodes[*vCurr-1] == uNodes[*uCurr]);
	*uCurr += vBigger;
	*vCurr += uBigger;
	return (uBigger + vBigger);
}

__device__ void intersectCount(const int uLength, const int vLength,
    int const * const __restrict__ uNodes, int const * const __restrict__ vNodes,
    int * const __restrict__ uCurr, int * const __restrict__ vCurr,
    int * const __restrict__ workIndex, int * const __restrict__ workPerThread,
    int * const __restrict__ triangles, int found)
{
  if((*uCurr < uLength) && (*vCurr < vLength)){
    int comp;
    while(*workIndex < *workPerThread){
		comp = uNodes[*uCurr] - vNodes[*vCurr];
		*triangles += (comp == 0);
		*uCurr += (comp <= 0);
		*vCurr += (comp >= 0);
		*workIndex += (comp == 0) + 1;

		if((*vCurr == vLength) || (*uCurr == uLength)){
			break;
		}
    }
    *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
  }
}


// u_len < v_len
__device__ int count_triangles(int u, int const * const __restrict__ u_nodes, int u_len,
    int v, int const * const __restrict__ v_nodes, int v_len, int threads_per_block,
    volatile int* __restrict__ firstFound, int tId)
{
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to Tersect - this number will be off by 1.
	  int work_per_thread, diag_id;
	  calcWorkPerThread(u_len, v_len, threads_per_block, tId, &work_per_thread, &diag_id);
	int triangles = 0;
	int work_index = 0,found=0;
	int u_min,u_max,v_min,v_max,u_curr,v_curr;

	firstFound[tId]=0;

	if(work_per_thread>0){
		// For the binary search, we are figuring out the initial poT of search.
		initialize(diag_id, u_len, v_len,&u_min, &u_max,&v_min, &v_max,&found);
    	u_curr = 0; v_curr = 0;

	    bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max, &v_min,
        &v_max, &u_curr, &v_curr);

    	int sum = fixThreadWorkEdges(u_len, v_len, &u_curr, &v_curr, u_nodes, v_nodes);
    	work_index += sum;
	    if(tId > 0)
	      firstFound[tId-1] = sum;
	    triangles += sum;
	    intersectCount(u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
	        &work_index, &work_per_thread, &triangles, firstFound[tId]);
	}
	return triangles;
}

__device__ void calcWorkPerBlock(const int numVertices,
    int * const __restrict__ outMpStart,
    int * const __restrict__ outMpEnd, int blockSize)
{
	int verticesPerMp = numVertices/gridDim.x;
	int remainderBlocks = numVertices % gridDim.x;
	int extraVertexBlocks = (blockIdx.x > remainderBlocks)? remainderBlocks:blockIdx.x;
	int regularVertexBlocks = (blockIdx.x > remainderBlocks)? blockIdx.x - remainderBlocks:0;

	int mpStart = ((verticesPerMp+1)*extraVertexBlocks) + (verticesPerMp*regularVertexBlocks);
	*outMpStart = mpStart;
	*outMpEnd = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}

__global__ void count_all_trianglesGPU (const int nv,
    int const * const __restrict__ d_off, int const * const __restrict__ d_ind,
    int * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter){

	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
	int tx = threadIdx.x;
 	int this_mp_start, this_mp_stop;

  const int blockSize = blockDim.x;
  calcWorkPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

  __shared__ int s_triangles[1024];
  __shared__ int firstFound[1024];

	int adj_offset=tx>>shifter;
	int* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (int src = this_mp_start; src < this_mp_stop; src++){
		int srcLen=d_off[src+1]-d_off[src];
	    int tCount = 0;
		for(int iter=d_off[src]+adj_offset; iter<d_off[src+1]; iter+=number_blocks){
			int dest = d_ind[iter];
			int destLen = d_off[dest+1]-d_off[dest];
			bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
			if(avoidCalc)
				continue;

	        bool sourceSmaller = (srcLen<destLen);
	        int small = sourceSmaller? src : dest;
	        int large = sourceSmaller? dest : src;
	        int small_len = sourceSmaller? srcLen : destLen;
	        int large_len = sourceSmaller? destLen : srcLen;

	        int const * const small_ptr = d_ind + d_off[small];
	        int const * const large_ptr = d_ind + d_off[large];
	        tCount += count_triangles(small, small_ptr, small_len,
						large,large_ptr, large_len,
						threads_per_block,firstFoundPos,
						tx%threads_per_block);
		}
		s_triangles[tx] = tCount;
		blockReduce(&outPutTriangles[src],s_triangles,blockSize);
	}
}


void callDeviceAllTrianglesCSR(const int nv,
    int const * const __restrict__ d_off, int const * const __restrict__ d_ind,
    int * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim){

	count_all_trianglesGPU<<<thread_blocks, blockdim>>>(nv,d_off, d_ind, outPutTriangles, threads_per_block,number_blocks,shifter);
}

