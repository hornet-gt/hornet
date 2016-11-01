#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include "cct.hpp"
#include "utils.hpp"

__device__ void conditionalWarpReduce(volatile triangle_t* sharedData,int blockSize,int dataLength){
  if(blockSize >= dataLength){
    if(threadIdx.x < (dataLength/2))
    {sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
    __syncthreads();
  }
}

__device__ void warpReduce(triangle_t* __restrict__ outDataPtr,
    volatile triangle_t* __restrict__ sharedData,int blockSize){
  conditionalWarpReduce(sharedData,blockSize,64);
  conditionalWarpReduce(sharedData,blockSize,32);
  conditionalWarpReduce(sharedData,blockSize,16);
  conditionalWarpReduce(sharedData,blockSize,8);
  conditionalWarpReduce(sharedData,blockSize,4);
  if(threadIdx.x == 0)
    {*outDataPtr= sharedData[0] + sharedData[1];}
  __syncthreads();
}

__device__ void conditionalReduce(volatile triangle_t* __restrict__ sharedData,int blockSize,int dataLength){
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

__device__ void blockReduce(triangle_t* __restrict__ outGlobalDataPtr,
    volatile triangle_t* __restrict__ sharedData,int blockSize){
  __syncthreads();
  conditionalReduce(sharedData,blockSize,1024);
  conditionalReduce(sharedData,blockSize,512);
  conditionalReduce(sharedData,blockSize,256);
  conditionalReduce(sharedData,blockSize,128);

  warpReduce(outGlobalDataPtr, sharedData, blockSize);
  __syncthreads();
}

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

template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ void intersectCount(const length_t uLength, const length_t vLength,
    vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
    length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
    int * const __restrict__ workIndex, int * const __restrict__ workPerThread,
    int * const __restrict__ triangles, int found, triangle_t * const __restrict__ outPutTriangles, 
    vertexId_t const * const __restrict__ uMask, vertexId_t const * const __restrict__ vMask)
{
  if((*uCurr < uLength) && (*vCurr < vLength)){
    int comp;
    int vmask;
    int umask;
    while(*workIndex < *workPerThread){
    	vmask = (vMasked) ? vMask[*vCurr] : 0;
        umask = (uMasked) ? uMask[*uCurr] : 0;
		comp = uNodes[*uCurr] - vNodes[*vCurr];
		*triangles += (comp == 0 && !umask && !vmask);
		if (upd3rdV && comp == 0 && !umask && !vmask)
			if (subtract) atomicSub(outPutTriangles + uNodes[*uCurr], 1);
			else atomicAdd(outPutTriangles + uNodes[*uCurr], 1);
		*uCurr += (comp <= 0 && !vmask) || umask;
		*vCurr += (comp >= 0 && !umask) || vmask;
		*workIndex += (comp == 0&& !umask && !vmask) + 1;

		if((*vCurr == vLength) || (*uCurr == uLength)){
			break;
		}
    }
    *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
  }
}


// u_len < v_len
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ triangle_t count_triangles(vertexId_t u, vertexId_t const * const __restrict__ u_nodes, length_t u_len,
    vertexId_t v, vertexId_t const * const __restrict__ v_nodes, length_t v_len, int threads_per_block,
    volatile vertexId_t* __restrict__ firstFound, int tId, triangle_t * const __restrict__ outPutTriangles,
    vertexId_t const * const __restrict__ uMask, vertexId_t const * const __restrict__ vMask)
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
	    intersectCount<uMasked, vMasked, subtract, upd3rdV>(
	    	u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
	        &work_index, &work_per_thread, &triangles, firstFound[tId], outPutTriangles, 
	        uMask, vMask);
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

	length_t *d_off = bud->getOffsets();
	vertexId_t * d_ind = bud->getDst();
	vertexId_t * d_seg = bud->getSrc();

	const int blockSize = blockDim.x;
	workPerBlock(batchSize, &this_mp_start, &this_mp_stop, blockSize);

	__shared__ vertexId_t firstFound[1024];

	length_t adj_offset=tx>>shifter;
	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (length_t edge = this_mp_start+adj_offset; edge < this_mp_stop; edge+=number_blocks){
		if (bud->getIndDuplicate()[edge]==1) // this means it's a duplicate edge
			continue;

		vertexId_t src = d_seg[edge];
		vertexId_t dest= d_ind[edge];

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

		triangle_t tCount = count_triangles<false, false, false, true>(
								small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block, outPutTriangles,
								NULL, NULL);

		atomicAdd(outPutTriangles + src, tCount);
		atomicAdd(outPutTriangles + dest, tCount);
		__syncthreads();
	}
}

__global__ void deviceBUThreeTriangles (BatchUpdateData *bud,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter)
{
	length_t batchsize = *(bud->getBatchSize());
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
	int tx = threadIdx.x;
 	length_t this_mp_start, this_mp_stop;

	length_t *d_off = bud->getOffsets();
	vertexId_t * d_ind = bud->getDst();
	vertexId_t * d_seg = bud->getSrc();

	const int blockSize = blockDim.x;
	workPerBlock(batchsize, &this_mp_start, &this_mp_stop, blockSize);

	__shared__ vertexId_t firstFound[1024];

	length_t adj_offset=tx>>shifter;
	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (length_t edge = this_mp_start+adj_offset; edge < this_mp_stop; edge+=number_blocks){
		if (bud->getIndDuplicate()[edge]) // this means it's a duplicate edge
			continue;
			
		vertexId_t src = d_seg[edge];
		vertexId_t dest= d_ind[edge];
		length_t srcLen= d_off[src+1] - d_off[src];
		length_t destLen=d_off[dest+1] - d_off[dest];

		bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
		if(avoidCalc)
			continue;

		bool sourceSmaller = (srcLen<destLen);
        vertexId_t small = sourceSmaller? src : dest;
        vertexId_t large = sourceSmaller? dest : src;
        length_t small_len = sourceSmaller? srcLen : destLen;
        length_t large_len = sourceSmaller? destLen : srcLen;

        vertexId_t const * const small_ptr = d_ind + d_off[small];
        vertexId_t const * const large_ptr = d_ind + d_off[large];
        vertexId_t const * const small_mask_ptr = bud->getIndDuplicate() + d_off[small];
        vertexId_t const * const large_mask_ptr = bud->getIndDuplicate() + d_off[large];

		triangle_t tCount = count_triangles<true, true, false, false>(
								small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block, outPutTriangles,
								small_mask_ptr, large_mask_ptr);

		atomicAdd(outPutTriangles + src, tCount);
		__syncthreads();
	}
}

__global__ void deviceBUTwoCUOneTriangles (BatchUpdateData *bud, cuStinger* custing,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter)
{
	length_t batchsize = *(bud->getBatchSize());
	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
	int tx = threadIdx.x;
 	vertexId_t this_mp_start, this_mp_stop;

	length_t *d_off = bud->getOffsets();
	vertexId_t * d_ind = bud->getDst();
	vertexId_t * d_seg = bud->getSrc();

	const int blockSize = blockDim.x;
	workPerBlock(batchsize, &this_mp_start, &this_mp_stop, blockSize);

	__shared__ vertexId_t firstFound[1024];

	length_t adj_offset=tx>>shifter;
	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
	for (length_t edge = this_mp_start+adj_offset; edge < this_mp_stop; edge+=number_blocks){
		if (bud->getIndDuplicate()[edge]) // this means it's a duplicate edge
			continue;
			
		vertexId_t src = bud->getSrc()[edge];
		vertexId_t dest= bud->getDst()[edge];
		length_t srcLen= d_off[src+1] - d_off[src];
		length_t destLen=custing->dVD->getUsed()[dest];

		bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
		if(avoidCalc)
			continue;

        vertexId_t const * const src_ptr = d_ind + d_off[src];
        vertexId_t const * const src_mask_ptr = bud->getIndDuplicate() + d_off[src];
        vertexId_t const * const dst_ptr = custing->dVD->getAdj()[dest]->dst;

		bool sourceSmaller = (srcLen<destLen);
        vertexId_t small = sourceSmaller? src : dest;
        vertexId_t large = sourceSmaller? dest : src;
        length_t small_len = sourceSmaller? srcLen : destLen;
        length_t large_len = sourceSmaller? destLen : srcLen;

        vertexId_t const * const small_ptr = sourceSmaller? src_ptr : dst_ptr;
        vertexId_t const * const small_mask_ptr = sourceSmaller? src_mask_ptr : NULL;
        vertexId_t const * const large_ptr = sourceSmaller? dst_ptr : src_ptr;
        vertexId_t const * const large_mask_ptr = sourceSmaller? NULL : src_mask_ptr;

		triangle_t tCount = (sourceSmaller)?
								count_triangles<true, false, true, true>(
								small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block, outPutTriangles,
								small_mask_ptr, large_mask_ptr):
								count_triangles<false, true, true, true>(
								small, small_ptr, small_len,
								large,large_ptr, large_len,
								threads_per_block,firstFoundPos,
								tx%threads_per_block, outPutTriangles,
								small_mask_ptr, large_mask_ptr )
							;

		atomicSub(outPutTriangles + src, tCount);
		atomicSub(outPutTriangles + dest, tCount);
		__syncthreads();
	}
}

void callDeviceNewTriangles(cuStinger& custing, BatchUpdate& bu, 
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim,
    triangle_t * const __restrict__ h_triangles, triangle_t * const __restrict__ h_triangles_t)
{
	cudaEvent_t ce_start,ce_stop;

	dim3 numBlocks(1, 1);

	length_t batchsize = *(bu.getHostBUD()->getBatchSize());
	length_t nv = *(bu.getHostBUD()->getNumVertices());

	numBlocks.x = ceil((float)(batchsize*threads_per_block)/(float)blockdim);

	// Calculate all new traingles regardless of repetition
		start_clock(ce_start, ce_stop);
		devicecuStingerNewTriangles<<<numBlocks, blockdim>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(), outPutTriangles, threads_per_block,number_blocks,shifter);
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));

	// Calculate triangles formed by only new edges
		start_clock(ce_start, ce_stop);
		deviceBUThreeTriangles<<<numBlocks,blockdim>>>(bu.getDeviceBUD()->devicePtr(), outPutTriangles, threads_per_block,number_blocks,shifter);
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));
	
	// Calculate triangles formed by two new edges
		start_clock(ce_start, ce_stop);
		deviceBUTwoCUOneTriangles<<<numBlocks,blockdim>>>(bu.getDeviceBUD()->devicePtr(),custing.devicePtr(), outPutTriangles, threads_per_block,number_blocks,shifter);
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));
}

__global__ void comparecus(cuStinger* cus1, cuStinger* cus2)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	length_t nv = cus1->nv;
	if (tid < nv)
	{
		vertexId_t * adj1 = cus1->dVD->adj[tid]->dst;
		vertexId_t * adj2 = cus2->dVD->adj[tid]->dst;
		length_t size1 = cus1->dVD->getUsed()[tid];
		length_t size2 = cus2->dVD->getUsed()[tid];
		if (size1 != size2)
		{
			printf("size mismatch %d %d\n", size1, size2);
		}
		for (int i = 0; i < size1; ++i)
		{
			if (adj1[i] != adj2[i])
			{
				printf("adj mismatch vertex %d, %d %d\n", tid, adj1[i], adj2[i]);
				for (int j = 0; j < size1; ++j)
				{
					printf("%d adj1 %d\n", tid, adj1[j]);
				}
				printf("%d ==\n", tid);
				for (int j = 0; j < size1; ++j)
				{
					printf("%d adj2 %d\n", tid, adj2[j]);
				}
			}
		}
	}
}

void compareCUS(cuStinger* cus1, cuStinger* cus2)
{
	length_t nv = cus1->nv;

	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	numBlocks.x = ceil((float)nv/(float)threads);
	comparecus<<<numBlocks, threadsPerBlock>>>(cus1->devicePtr(),cus2->devicePtr());
}
