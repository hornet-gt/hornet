
#include <cuda.h>
#include <cuda_runtime.h>


#include "Static/TriangleCounting/triangle.cuh"

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

__device__ __forceinline__
void intersectCount(const custinger::cuStingerDevice& custinger,
		degree_t uLength, degree_t vLength,
		const vid_t*  __restrict__ uNodes,
		const vid_t*  __restrict__ vNodes,
		vid_t*  __restrict__ uCurr,
		vid_t*  __restrict__ vCurr,
		int*    __restrict__ workIndex,
		const int*    __restrict__ workPerThread,
		triangle_t*    __restrict__ triangles,
		int found,
		triangle_t*  __restrict__ outPutTriangles,
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
__device__ __forceinline__
triangle_t count_triangles(const custinger::cuStingerDevice& custinger,
                           vid_t u,
                           const vid_t* __restrict__ u_nodes,
                           degree_t u_len,
                           vid_t v,
                           const vid_t* __restrict__ v_nodes,
                           degree_t v_len,
                           int   threads_per_block,
                           volatile triangle_t* __restrict__ firstFound,
                           int    tId,
                           triangle_t* __restrict__ outPutTriangles,
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
        intersectCount
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
                           triangle_t* __restrict__ outPutTriangles,
                           int threads_per_block,
                           int number_blocks,
                           int shifter,
                           TriangleData* __restrict__ devData) {
    vid_t nv = custinger.nV();
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
		Vertex vertex = custinger.vertex(src);
        vid_t srcLen = vertex.degree();

        // triangle_t tCount = 0;
        for(int k = adj_offset; k < srcLen; k += number_blocks) {
            //vid_t  dest = custinger->dVD->getAdj()[src]->dst[k];
            // vid_t dest = vertex.edge(k).dst();
            vid_t dest = vertex.edge(k).dst_id();
            //int destLen = custinger->dVD->getUsed()[dest];
            degree_t destLen = custinger.vertex(dest).degree();
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

            // const vid_t* small_ptr = Vertex(custinger, small).neighbor_ptr();
            // const vid_t* large_ptr = Vertex(custinger, large).neighbor_ptr();

            const vid_t* small_ptr = custinger.vertex(small).neighbor_ptr();
            const vid_t* large_ptr = custinger.vertex(large).neighbor_ptr();

            triangle_t triFound = count_triangles
                (custinger, small, small_ptr, small_len, large, large_ptr,
                 large_len, threads_per_block, (triangle_t*)firstFoundPos,
                 tx % threads_per_block, outPutTriangles,
                 nullptr, nullptr, 1, src, dest);
            // tCount += triFound;
						
            atomicAdd(outPutTriangles+src,triFound);
            atomicAdd(outPutTriangles+dest,triFound);
						// printf("Need to add an atomic add here");

        }
    //    s_triangles[tx] = tCount;
    //    blockReduce(&outPutTriangles[src],s_triangles,blockSize);
    }
}

void staticTriangleCounting(cuStinger& custinger,
                        triangle_t* __restrict__ outPutTriangles,
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

// -----------------------
// -----------------------
// -----------------------
// -----------------------
// The above functions are responsible for doing the actual triangle counting.
// The functions below are the StaticAlgorithm functions used for running the algorithm.
// -----------------------
// -----------------------
// -----------------------
// -----------------------

TriangleCounting::TriangleCounting(custinger::cuStinger& custinger) :
                                       StaticAlgorithm(custinger),
																			 hostTriangleData(custinger){
	deviceTriangleData = register_data(hostTriangleData);
	memReleased = true;
}

TriangleCounting::~TriangleCounting(){
	release();
}

void TriangleCounting::reset(){
	forAllnumV<triangle_operators::init>(custinger,deviceTriangleData);
}

void TriangleCounting::run(){

	staticTriangleCounting(custinger,
                        hostTriangleData.triPerVertex,
                        hostTriangleData.threadsPerIntersection,
                        hostTriangleData.numberInterPerBlock,
                        hostTriangleData.logThreadsPerInter,
                        hostTriangleData.threadBlocks,
                        hostTriangleData.blockSize,
                        deviceTriangleData);
}

void TriangleCounting::release(){
	if(memReleased)
		return;
	memReleased=true;
	gpu::free(hostTriangleData.triPerVertex);

}


void TriangleCounting::setInitParameters(int threadBlocks, int blockSize, int threadsPerIntersection){
	hostTriangleData.threadBlocks					= threadBlocks;
	hostTriangleData.blockSize    					= blockSize;

	if(hostTriangleData.blockSize%32 != 0){
		printf("The block size has to be a multiple of 32\n");
		printf("The block size has to be a reduced to the closet multiple of 32\n");
		hostTriangleData.blockSize = (hostTriangleData.blockSize/32)*32;
	}
	if(hostTriangleData.blockSize < 0){
		printf("The block size has to be a positive numbe\n");
		exit(0);
	}
	
	hostTriangleData.threadsPerIntersection = threadsPerIntersection;
	if(hostTriangleData.threadsPerIntersection <= 0 || hostTriangleData.threadsPerIntersection >32 ){
		printf("Threads per intersection have to be a power of two between 1 and 32\n");
		exit(0);
	}
	int temp = hostTriangleData.threadsPerIntersection,logtemp=0;
	while (temp>>=1) ++logtemp;
	hostTriangleData.logThreadsPerInter			= logtemp;
	hostTriangleData.numberInterPerBlock=hostTriangleData.blockSize/hostTriangleData.threadsPerIntersection;
}


void TriangleCounting::init(){
	memReleased=false;
	gpu::allocate(hostTriangleData.triPerVertex, hostTriangleData.nv+10);
	syncDeviceWithHost();
	reset();
}

triangle_t TriangleCounting::countTriangles(){
 //    triangle_t* outputArray = (triangle_t*)malloc((hostTriangleData.nv+2)*sizeof(triangle_t));
 //    cudaMemcpy(outputArray,hostTriangleData.triPerVertex,(hostTriangleData.nv+2)*sizeof(triangle_t),cudaMemcpyDeviceToHost);
 //    triangle_t sum=0;
 //    for(int i=0; i<(hostTriangleData.nv); i++){
 //        // printf("%d %ld\n", i,outputArray[i]);
 //        sum+=outputArray[i];
 //    }
	// // // free(outputArray);
	triangle_t sum=gpu::reduce(hostTriangleData.triPerVertex, hostTriangleData.nv+1);

    return sum;
}
} // namespace custinger_alg

