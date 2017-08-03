// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <inttypes.h>

// #include "static_triangle_counting/cct.hpp"
 
// __device__ void conditionalWarpReduceIP(volatile triangle_t* sharedData,int blockSize,int dataLength){
//   if(blockSize >= dataLength){
//     if(threadIdx.x < (dataLength/2))
//     {sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
//     __syncthreads();
//   }
// }

// __device__ void warpReduceIP(triangle_t* __restrict__ outDataPtr,
//     volatile triangle_t* __restrict__ sharedData,int blockSize){
//   conditionalWarpReduceIP(sharedData,blockSize,64);
//   conditionalWarpReduceIP(sharedData,blockSize,32);
//   conditionalWarpReduceIP(sharedData,blockSize,16);
//   conditionalWarpReduceIP(sharedData,blockSize,8);
//   conditionalWarpReduceIP(sharedData,blockSize,4);
//   if(threadIdx.x == 0)
//     {*outDataPtr= sharedData[0] + sharedData[1];}
//   __syncthreads();
// }

// __device__ void conditionalReduceIP(volatile triangle_t* __restrict__ sharedData,int blockSize,int dataLength){
// 	if(blockSize >= dataLength){
// 		if(threadIdx.x < (dataLength/2))
// 		{sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
// 		__syncthreads();
// 	}
// 	if((blockSize < dataLength) && (blockSize > (dataLength/2))){
// 		if(threadIdx.x+(dataLength/2) < blockSize){
// 			sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];
// 		}
// 		__syncthreads();
// 	}
// }

// __device__ void blockReduceIP(triangle_t* __restrict__ outGlobalDataPtr,
//     volatile triangle_t* __restrict__ sharedData,int blockSize){
//   __syncthreads();
//   conditionalReduceIP(sharedData,blockSize,1024);
//   conditionalReduceIP(sharedData,blockSize,512);
//   conditionalReduceIP(sharedData,blockSize,256);
//   conditionalReduceIP(sharedData,blockSize,128);

//   warpReduceIP(outGlobalDataPtr, sharedData, blockSize);
//   __syncthreads();
// }

// __device__ void initializeIP(const vertexId_t diag_id, const length_t u_len, length_t v_len,
//     length_t* const __restrict__ u_min, length_t* const __restrict__ u_max,
//     length_t* const __restrict__ v_min, length_t* const __restrict__ v_max,
//     int* const __restrict__ found)
// {
// 	if (diag_id == 0){
// 		*u_min=*u_max=*v_min=*v_max=0;
// 		*found=1;
// 	}
// 	else if (diag_id < u_len){
// 		*u_min=0; *u_max=diag_id;
// 		*v_max=diag_id;*v_min=0;
// 	}
// 	else if (diag_id < v_len){
// 		*u_min=0; *u_max=u_len;
// 		*v_max=diag_id;*v_min=diag_id-u_len;
// 	}
// 	else{
// 		*u_min=diag_id-v_len; *u_max=u_len;
// 		*v_min=diag_id-u_len; *v_max=v_len;
// 	}
// }

// __device__ void workPerThreadIP(const length_t uLength, const length_t vLength, 
// 	const int threadsPerIntersection, const int threadId,
//     int * const __restrict__ outWorkPerThread, int * const __restrict__ outDiagonalId){
//   int totalWork = uLength + vLength;
//   int remainderWork = totalWork%threadsPerIntersection;
//   int workPerThread = totalWork/threadsPerIntersection;

//   int longDiagonals  = (threadId > remainderWork) ? remainderWork:threadId;
//   int shortDiagonals = (threadId > remainderWork) ? (threadId - remainderWork):0;

//   *outDiagonalId = ((workPerThread+1)*longDiagonals) + (workPerThread*shortDiagonals);
//   *outWorkPerThread = workPerThread + (threadId < remainderWork);
// }

// __device__ void bSearchIP(unsigned int found, const vertexId_t diagonalId,
//     vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
//     length_t const * const __restrict__ uLength, 
//     length_t * const __restrict__ outUMin, length_t * const __restrict__ outUMax,
//     length_t * const __restrict__ outVMin, length_t * const __restrict__ outVMax,    
//     length_t * const __restrict__ outUCurr,
//     length_t * const __restrict__ outVCurr){
//   	length_t length;
	
// 	while(!found) {
// 	    *outUCurr = (*outUMin + *outUMax)>>1;
// 	    *outVCurr = diagonalId - *outUCurr;
// 	    if(*outVCurr >= *outVMax){
// 			length = *outUMax - *outUMin;
// 			if(length == 1){
// 				found = 1;
// 				continue;
// 			}
// 	    }

// 	    unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr-1];
// 	    unsigned int comp2 = uNodes[*outUCurr-1] > vNodes[*outVCurr];
// 	    if(comp1 && !comp2){
// 			found = 1;
// 	    }
// 	    else if(comp1){
// 	      *outVMin = *outVCurr;
// 	      *outUMax = *outUCurr;
// 	    }
// 	    else{
// 	      *outVMax = *outVCurr;
// 	      *outUMin = *outUCurr;
// 	    }
//   	}

// 	if((*outVCurr >= *outVMax) && (length == 1) && (*outVCurr > 0) &&
// 	(*outUCurr > 0) && (*outUCurr < (*uLength - 1))){
// 		unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
// 		unsigned int comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
// 		if(!comp1 && !comp2){(*outUCurr)++; (*outVCurr)--;}
// 	}
// }

// __device__ int fixStartPointIP(const length_t uLength, const length_t vLength,
//     length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
//     vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes){
	
// 	unsigned int uBigger = (*uCurr > 0) && (*vCurr < vLength) && (uNodes[*uCurr-1] == vNodes[*vCurr]);
// 	unsigned int vBigger = (*vCurr > 0) && (*uCurr < uLength) && (vNodes[*vCurr-1] == uNodes[*uCurr]);
// 	*uCurr += vBigger;
// 	*vCurr += uBigger;
// 	return (uBigger + vBigger);
// }

// __device__ void intersectPath(const length_t uLength, const length_t vLength,
//     vertexId_t const * const __restrict__ uNodes, vertexId_t const * const __restrict__ vNodes,
//     length_t * const __restrict__ uCurr, length_t * const __restrict__ vCurr,
//     int * const __restrict__ workIndex, int * const __restrict__ workPerThread,
//     int * const __restrict__ triangles, int found)
// {
//   if((*uCurr < uLength) && (*vCurr < vLength)){
//     int comp;
//     while(*workIndex < *workPerThread){
// 		comp = uNodes[*uCurr] - vNodes[*vCurr];
// 		*triangles += (comp == 0);
// 		*uCurr += (comp <= 0);
// 		*vCurr += (comp >= 0);
// 		*workIndex += (comp == 0) + 1;

// 		if((*vCurr == vLength) || (*uCurr == uLength)){
// 			break;
// 		}
//     }
//     *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
//   }
// }


// // u_len < v_len
// __device__ triangle_t singleIntersection(vertexId_t u, vertexId_t const * const __restrict__ u_nodes, length_t u_len,
//     vertexId_t v, vertexId_t const * const __restrict__ v_nodes, length_t v_len, int threads_per_block,
//     volatile vertexId_t* __restrict__ firstFound, int tId)
// {
// 	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to Tersect - this number will be off by 1.
// 	int work_per_thread, diag_id;
// 	workPerThreadIP(u_len, v_len, threads_per_block, tId, &work_per_thread, &diag_id);
// 	triangle_t triangles = 0;
// 	int work_index = 0,found=0;
// 	length_t u_min,u_max,v_min,v_max,u_curr,v_curr;

// 	firstFound[tId]=0;

// 	if(work_per_thread>0){
// 		// For the binary search, we are figuring out the initial poT of search.
// 		initializeIP(diag_id, u_len, v_len,&u_min, &u_max,&v_min, &v_max,&found);
//     	u_curr = 0; v_curr = 0;
// 	    bSearchIP(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max, &v_min,
//         &v_max, &u_curr, &v_curr);

//     	int sum = fixStartPointIP(u_len, v_len, &u_curr, &v_curr, u_nodes, v_nodes);
//     	work_index += sum;
// 	    if(tId > 0)
// 	      firstFound[tId-1] = sum;
// 	    triangles += sum;
// 	    intersectPath(u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
// 	        &work_index, &work_per_thread, &triangles, firstFound[tId]);
// 	}
// 	return triangles;
// }

// __device__ void workPerBlockIP(const vertexId_t numVertices,
//     vertexId_t * const __restrict__ outMpStart,
//     vertexId_t * const __restrict__ outMpEnd, int blockSize)
// {
// 	vertexId_t verticesPerMp = numVertices/gridDim.x;
// 	vertexId_t remainderBlocks = numVertices % gridDim.x;
// 	vertexId_t extraVertexBlocks = (blockIdx.x > remainderBlocks)? remainderBlocks:blockIdx.x;
// 	vertexId_t regularVertexBlocks = (blockIdx.x > remainderBlocks)? blockIdx.x - remainderBlocks:0;

// 	vertexId_t mpStart = ((verticesPerMp+1)*extraVertexBlocks) + (verticesPerMp*regularVertexBlocks);
// 	*outMpStart = mpStart;
// 	*outMpEnd = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
// }


// __global__ void devicecuStingerAllTriangles(cuStinger* custing,
//     triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
//     const int number_blocks, const int shifter)
// {
// 	vertexId_t nv = custing->nv;
// 	// Partitioning the work to the multiple thread of a single GPU processor. The threads should get a near equal number of the elements to intersect - this number will be off by no more than one.
// 	int tx = threadIdx.x;
//  	vertexId_t this_mp_start, this_mp_stop;

// 	const int blockSize = blockDim.x;
// 	workPerBlockIP(nv, &this_mp_start, &this_mp_stop, blockSize);

// 	__shared__ triangle_t  s_triangles[1024];
// 	__shared__ vertexId_t firstFound[1024];

// 	length_t adj_offset=tx>>shifter;
// 	length_t* firstFoundPos=firstFound + (adj_offset<<shifter);
// 	for (vertexId_t src = this_mp_start; src < this_mp_stop; src++){
// 		// int srcLen=d_off[src+1]-d_off[src];
// 		length_t srcLen=custing->dVD->getUsed()[src];
// 	    triangle_t tCount = 0;	    
// 		// for(int iter=d_off[src]+adj_offset; iter<d_off[src+1]; iter+=number_blocks){
// 		for(int k=adj_offset; k<srcLen; k+=number_blocks){
// 			// int dest = d_ind[k];
// 			vertexId_t dest = custing->dVD->getAdj()[src]->dst[k];
// 			// int destLen = d_off[dest+1]-d_off[dest];
// 			int destLen=custing->dVD->getUsed()[dest];

// 			// if (dest<src) 
// 			// 	continue;

// 			bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
// 			if(avoidCalc)
// 				continue;

// 	        bool sourceSmaller = (srcLen<destLen);
// 	        vertexId_t small = sourceSmaller? src : dest;
// 	        vertexId_t large = sourceSmaller? dest : src;
// 	        length_t small_len = sourceSmaller? srcLen : destLen;
// 	        length_t large_len = sourceSmaller? destLen : srcLen;


// 	        // int const * const small_ptr = d_ind + d_off[small];
// 	        // int const * const large_ptr = d_ind + d_off[large];
// 	        const vertexId_t* small_ptr = custing->dVD->getAdj()[small]->dst;
// 	        const vertexId_t* large_ptr = custing->dVD->getAdj()[large]->dst;
// 	        tCount += singleIntersection(small, small_ptr, small_len,
// 						large,large_ptr, large_len,
// 						threads_per_block,firstFoundPos,
// 						tx%threads_per_block);
// 		}
// 		s_triangles[tx] = tCount;
// 		blockReduceIP(&outPutTriangles[src],s_triangles,blockSize);
// 	}
// }

// void callDeviceAllTriangles(cuStinger& custing,
//     triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
//     const int number_blocks, const int shifter, const int thread_blocks, const int blockdim){

// 	devicecuStingerAllTriangles<<<thread_blocks, blockdim>>>(custing.devicePtr(), outPutTriangles, threads_per_block,number_blocks,shifter);
// }

#include "Static/TriangleCounting/triangle.cuh"

using namespace custinger;
using namespace custinger_alg;


namespace custinger_alg {

__device__ __forceinline__
void initialize(degree_t diag_id,
                degree_t u_len,
                degree_t v_len,
                vid_t* __restrict__ u_min,
                vid_t* __restrict__ u_max,
                vid_t* __restrict__ v_min,
                vid_t* __restrict__ v_max,
                int*   __restrict__ found) {
    if (diag_id == 0) {
        *u_min = *u_max = *v_min = *v_max = 0;
        *found = 1;
    }
    else if (diag_id < u_len) {
        *u_min = 0;
        *u_max = diag_id;
        *v_max = diag_id;
        *v_min = 0;
    }
    else if (diag_id < v_len) {
        *u_min = 0;
        *u_max = u_len;
        *v_max = diag_id;
        *v_min = diag_id - u_len;
    }
    else {
        *u_min = diag_id - v_len;
        *u_max = u_len;
        *v_min = diag_id - u_len;
        *v_max = v_len;
    }
}

__device__ __forceinline__
void workPerThread(degree_t uLength,
                   degree_t vLength,
                   int threadsPerIntersection,
                   int threadId,
                   int* __restrict__ outWorkPerThread,
                   int* __restrict__ outDiagonalId) {
  int      totalWork = uLength + vLength;
  int  remainderWork = totalWork % threadsPerIntersection;
  int  workPerThread = totalWork / threadsPerIntersection;

  int longDiagonals  = threadId > remainderWork ? remainderWork : threadId;
  int shortDiagonals = threadId > remainderWork ? threadId - remainderWork : 0;

  *outDiagonalId     = (workPerThread + 1) * longDiagonals +
                        workPerThread * shortDiagonals;
  *outWorkPerThread  = workPerThread + (threadId < remainderWork);
}

__device__ __forceinline__
void bSearch(unsigned found,
             degree_t    diagonalId,
             const vid_t*  __restrict__ uNodes,
             const vid_t*  __restrict__ vNodes,
             const degree_t*  __restrict__ uLength,
             vid_t* __restrict__ outUMin,
             vid_t* __restrict__ outUMax,
             vid_t* __restrict__ outVMin,
             vid_t* __restrict__ outVMax,
             vid_t* __restrict__ outUCurr,
             vid_t* __restrict__ outVCurr) {
	vid_t length;
	while (!found){
		*outUCurr = (*outUMin + *outUMax) >> 1;
		*outVCurr = diagonalId - *outUCurr;
		if (*outVCurr >= *outVMax){
			length = *outUMax - *outUMin;
			if (length == 1){
				found = 1;
				continue;
			}
		}

		unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
		unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
		if (comp1 && !comp2)
			found = 1;
		else if (comp1){
			*outVMin = *outVCurr;
			*outUMax = *outUCurr;
		}
		else{
			*outVMax = *outVCurr;
			*outUMin = *outUCurr;
		}
	}

	if (*outVCurr >= *outVMax && length == 1 && *outVCurr > 0 &&
			*outUCurr > 0 && *outUCurr < *uLength - 1)
	{
		unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
		unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
		if (!comp1 && !comp2)
		{
			(*outUCurr)++;
			(*outVCurr)--;
		}
	}
}

__device__ __forceinline__
int fixStartPoint(degree_t uLength, degree_t vLength,
                  vid_t* __restrict__ uCurr,
                  vid_t* __restrict__ vCurr,
                  const vid_t* __restrict__ uNodes,
                  const vid_t* __restrict__ vNodes) {

    unsigned uBigger = (*uCurr > 0) && (*vCurr < vLength) &&
                       (uNodes[*uCurr - 1] == vNodes[*vCurr]);
    unsigned vBigger = (*vCurr > 0) && (*uCurr < uLength) &&
                       (vNodes[*vCurr - 1] == uNodes[*uCurr]);
    *uCurr += vBigger;
    *vCurr += uBigger;
    return uBigger + vBigger;
}

__device__ __forceinline__
void indexBinarySearch(vid_t* data, vid_t arrLen, vid_t key, int& pos) {
	int low = 0;
	int high = arrLen - 1;
	while (high >= low)
	{
		int middle = (low + high) / 2;
		if (data[middle] == key)
		{
			pos = middle;
			return;
		}
		if (data[middle] < key)
			low = middle + 1;
		if (data[middle] > key)
			high = middle - 1;
	}
}

template<bool subtract, bool upd3rdV>
__device__ __forceinline__
void intersectCount(const custinger::cuStingerDevice& custinger,
		degree_t uLength, degree_t vLength,
		const vid_t*  __restrict__ uNodes,
		const vid_t*  __restrict__ vNodes,
		vid_t*  __restrict__ uCurr,
		vid_t*  __restrict__ vCurr,
		int*    __restrict__ workIndex,
		const int*    __restrict__ workPerThread,
		int*    __restrict__ triangles,
		int found,
		const triangle_t*  __restrict__ outPutTriangles,
		vid_t src, vid_t dest,
    vid_t u, vid_t v) {
	if (*uCurr < uLength && *vCurr < vLength) {
		int comp;
		int vmask;
		int umask;
		while (*workIndex < *workPerThread)
		{
			vmask = umask = 0;
			comp = uNodes[*uCurr] - vNodes[*vCurr];

			*triangles += (comp == 0);

			*uCurr += (comp <= 0 && !vmask) || umask;
			*vCurr += (comp >= 0 && !umask) || vmask;
			*workIndex += (comp == 0 && !umask && !vmask) + 1;

			if (*vCurr >= vLength || *uCurr >= uLength)
				break;
		}
		*triangles -= ((comp == 0) && (*workIndex > *workPerThread) && found);
	}
}

// u_len < v_len
template <bool subtract, bool upd3rdV>
__device__ __forceinline__
triangle_t count_triangles(const custinger::cuStingerDevice& custinger,
                           vid_t u,
                           const vid_t* __restrict__ u_nodes,
                           degree_t u_len,
                           vid_t v,
                           const vid_t* __restrict__ v_nodes,
                           degree_t v_len,
                           int   threads_per_block,
                           volatile vid_t* __restrict__ firstFound,
                           int    tId,
                           const triangle_t* __restrict__ outPutTriangles,
                           const vid_t*      __restrict__ uMask,
                           const vid_t*      __restrict__ vMask,
                           triangle_t multiplier,
                           vid_t      src,
                           vid_t      dest) {

    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //Tersect - this number will be off by 1.
    int work_per_thread, diag_id;
    workPerThread(u_len, v_len, threads_per_block, tId,
                  &work_per_thread, &diag_id);
    triangle_t triangles = 0;
    int       work_index = 0;
    int            found = 0;
    vid_t u_min, u_max, v_min, v_max, u_curr, v_curr;

    firstFound[tId] = 0;

    if (work_per_thread > 0) {
        // For the binary search, we are figuring out the initial poT of search.
        initialize(diag_id, u_len, v_len, &u_min, &u_max,
                   &v_min, &v_max, &found);
        u_curr = 0;
        v_curr = 0;

        bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max,
                &v_min, &v_max, &u_curr, &v_curr);

        int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr,
                                u_nodes, v_nodes);
        work_index += sum;
        if (tId > 0)
           firstFound[tId - 1] = sum;
        triangles += sum;
        intersectCount<subtract, upd3rdV>
            (custinger, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
            &work_index, &work_per_thread, &triangles, firstFound[tId],
            outPutTriangles, src, dest, u, v);
    }
    return triangles;
}

__device__ __forceinline__
void workPerBlock(vid_t numVertices,
                  vid_t* __restrict__ outMpStart,
                  vid_t* __restrict__ outMpEnd,
                  int blockSize) {
    vid_t       verticesPerMp = numVertices / gridDim.x;
    vid_t     remainderBlocks = numVertices % gridDim.x;
    vid_t   extraVertexBlocks = (blockIdx.x > remainderBlocks) ? remainderBlocks
                                                               : blockIdx.x;
    vid_t regularVertexBlocks = (blockIdx.x > remainderBlocks) ?
                                    blockIdx.x - remainderBlocks : 0;

    vid_t mpStart = (verticesPerMp + 1) * extraVertexBlocks +
                     verticesPerMp * regularVertexBlocks;
    *outMpStart   = mpStart;
    *outMpEnd     = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}

__global__
void devicecuStaticTriangleCounting(custinger::cuStingerDevice custinger,
                           const triangle_t* __restrict__ outPutTriangles,
                           int threads_per_block,
                           int number_blocks,
                           int shifter,
                           TriangleData* __restrict__ devData) {
    vid_t nv = custinger.nV;
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements
    //to intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vid_t this_mp_start, this_mp_stop;

    const int blockSize = blockDim.x;
    workPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

    //__shared__ triangle_t s_triangles[1024];
    __shared__ vid_t      firstFound[1024];

    vid_t     adj_offset = tx >> shifter;
    vid_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vid_t src = this_mp_start; src < this_mp_stop; src++) {
        //vid_t      srcLen = custinger->dVD->getUsed()[src];
        Vertex vertex(custinger, src);
        vid_t srcLen = vertex.degree();

        triangle_t tCount = 0;
        for(int k = adj_offset; k < srcLen; k += number_blocks) {
            //vid_t  dest = custinger->dVD->getAdj()[src]->dst[k];
            // vid_t dest = vertex.edge(k).dst();
            vid_t dest = vertex.edge(k).dst_id();
            //int destLen = custinger->dVD->getUsed()[dest];
            degree_t destLen = Vertex(custinger, dest).degree();

            if (dest < src) //opt
                continue;   //opt

            bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
            if (avoidCalc)
                continue;

            bool sourceSmaller = srcLen < destLen;
            vid_t        small = sourceSmaller ? src : dest;
            vid_t        large = sourceSmaller ? dest : src;
            degree_t    small_len = sourceSmaller ? srcLen : destLen;
            degree_t    large_len = sourceSmaller ? destLen : srcLen;

            const vid_t* small_ptr = Vertex(custinger, small).neighbor_ptr();
            const vid_t* large_ptr = Vertex(custinger, large).neighbor_ptr();

            triangle_t triFound = count_triangles<false, false>
                (custinger, small, small_ptr, small_len, large, large_ptr,
                 large_len, threads_per_block, firstFoundPos,
                 tx % threads_per_block, outPutTriangles,
                 nullptr, nullptr, 1, src, dest);
            tCount += triFound;
						
						printf("Need to add an atomic add here");

        }
    //    s_triangles[tx] = tCount;
    //    blockReduce(&outPutTriangles[src],s_triangles,blockSize);
    }
}

void staticTriangleCounting(cuStinger& custinger,
                        const triangle_t* __restrict__ outPutTriangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        TriangleData* __restrict__ devData) {

    devicecuStaticTriangleCounting <<< thread_blocks, blockdim >>>
        (custinger.device_side(), outPutTriangles, threads_per_block,
         number_blocks, shifter, devData);
}

} // namespace custinger_alg

