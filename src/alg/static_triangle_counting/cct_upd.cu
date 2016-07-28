#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include "cct.hpp"
 
__device__ void initialize(const vertexId_t diag_id, const length_t u_len, length_t v_len,
    length_t* const __restrict__ u_min, length_t* const __restrict__ u_max,
    length_t* const __restrict__ v_min, length_t* const __restrict__ v_max,
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

__device__ void workPerThread(const length_t uLength, const length_t vLength, 
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

__device__ void bSearch(unsigned int found, const vertexId_t diagonalId,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
    length_t const * const __restrict__ uLength, 
    length_t * const __restrict__ outUMin, length_t * const __restrict__ outUMax,
    length_t * const __restrict__ outVMin, length_t * const __restrict__ outVMax,    
    length_t * const __restrict__ outUCurr,
    length_t * const __restrict__ outVCurr){
  	length_t length;
	
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

__device__ int fixStartPoint(const length_t uLength, const length_t vLength,
    length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes){
	
	unsigned int uBigger = (*uCurr > 0) && (*vCurr < vLength) && (uNodes[*uCurr-1] == vNodes[*vCurr]);
	unsigned int vBigger = (*vCurr > 0) && (*uCurr < uLength) && (vNodes[*vCurr-1] == uNodes[*uCurr]);
	*uCurr += vBigger;
	*vCurr += uBigger;
	return (uBigger + vBigger);
}

__device__ void intersectCount(const length_t uLength, const length_t vLength,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
    length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
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
__device__ triangle_t count_triangles(vertexId_t u, vertexId_t const * const __restrict__ u_nodes, length_t u_len,
    vertexId_t v, vertexId_t const * const __restrict__ v_nodes, length_t v_len, int threads_per_block,
    volatile vertexId_t* __restrict__ firstFound, int tId)
{
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to Tersect - this number will be off by 1.
	int work_per_thread, diag_id;
	workPerThread(u_len, v_len, threads_per_block, tId, &work_per_thread, &diag_id);
	triangle_t triangles = 0;
	int work_index = 0,found=0;
	length_t u_min,u_max,v_min,v_max,u_curr,v_curr;

	firstFound[tId]=0;

	if(work_per_thread>0){
		// For the binary search, we are figuring out the initial poT of search.
		initialize(diag_id, u_len, v_len,&u_min, &u_max,&v_min, &v_max,&found);
    	u_curr = 0; v_curr = 0;

	    bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max, &v_min,
        &v_max, &u_curr, &v_curr);

    	int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr, u_nodes, v_nodes);
    	work_index += sum;
	    if(tId > 0)
	      firstFound[tId-1] = sum;
	    triangles += sum;
	    intersectCount(u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
	        &work_index, &work_per_thread, &triangles, firstFound[tId]);
	}
	return triangles;
}

__device__ void workPerBlock(const length_t numVertices,
    length_t * const __restrict__ outMpStart,
    length_t * const __restrict__ outMpEnd, int blockSize)
{
	length_t verticesPerMp = numVertices/gridDim.x;
	length_t remainderBlocks = numVertices % gridDim.x;
	length_t extraVertexBlocks = (blockIdx.x > remainderBlocks)? remainderBlocks:blockIdx.x;
	length_t regularVertexBlocks = (blockIdx.x > remainderBlocks)? blockIdx.x - remainderBlocks:0;

	length_t mpStart = ((verticesPerMp+1)*extraVertexBlocks) + (verticesPerMp*regularVertexBlocks);
	*outMpStart = mpStart;
	*outMpEnd = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}


__global__ void devicecuStingerNewTriangles(cuStinger* custing, BatchUpdateData *bud,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter)
{
	length_t batchSize = *(bud->getBatchSize());
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
	int tx = threadIdx.x;
 	length_t this_mp_start, this_mp_stop;

	const int blockSize = blockDim.x;
	workPerBlock(batchSize, &this_mp_start, &this_mp_stop, blockSize);

	__shared__ vertexId_t firstFound[1024];

	length_t adj_offset=tx>>shifter;
	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (length_t edge = this_mp_start+adj_offset; edge < this_mp_stop; edge+=number_blocks){
		vertexId_t src = bud->getSrc()[edge];
		vertexId_t dest= bud->getDst()[edge];
		length_t srcLen=custing->dVD->getUsed()[src];
		length_t destLen=custing->dVD->getUsed()[dest];

		bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
		if(avoidCalc)
			continue;

		bool sourceSmaller = (srcLen<destLen);
        vertexId_t small = sourceSmaller? src : dest;
        vertexId_t large = sourceSmaller? dest : src;
        length_t small_len = sourceSmaller? srcLen : destLen;
        length_t large_len = sourceSmaller? destLen : srcLen;

        const vertexId_t* small_ptr = custing->dVD->getAdj()[small]->dst;
        const vertexId_t* large_ptr = custing->dVD->getAdj()[large]->dst;

		triangle_t tCount = count_triangles(small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block);

		atomicAdd(outPutTriangles + src, tCount);
	}
}

void callDeviceNewTriangles(cuStinger& custing, BatchUpdate& bu,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim){

	devicecuStingerNewTriangles<<<thread_blocks, blockdim>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(), outPutTriangles, threads_per_block,number_blocks,shifter);
}



