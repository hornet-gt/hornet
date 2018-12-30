
#include <cuda.h>
#include <cuda_runtime.h>

/*
Please cite:
* O. Green, P. Yalamanchili ,L.M. Munguia, “Fast Triangle Counting on GPU”, 
Irregular Applications: Architectures and Algorithms (IA3), New Orleans, Louisiana, 2014 
* O. Green, R. McColl, D. Bader, "GPU Merge Path - A GPU Merging Algorithm", 
ACM 26th International Conference on Supercomputing, Venice, Italy, 2012
*/

#include "Static/TriangleCounting/triangle.cuh"

using namespace hornets_nest;

namespace hornets_nest {

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

template<typename HornetDevice>
__device__ __forceinline__
void intersectCount(const HornetDevice& hornet,
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
template<typename HornetDevice>
__device__ __forceinline__
triangle_t count_triangles(const HornetDevice& hornet,
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
            (hornet, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
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

template<typename HornetDevice>
__global__
void devicecuStaticTriangleCounting(HornetDevice hornet,
                           triangle_t* __restrict__ outPutTriangles,
                           int threads_per_block,
                           int number_blocks,
                           int shifter,
                           int cutoff,
                           HostDeviceVar<TriangleData> hd_data) {
    TriangleData* __restrict__ devData = hd_data.ptr();
    vid_t nv = hornet.nV();
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements
    //to intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vid_t this_mp_start, this_mp_stop;

    const int blockSize = blockDim.x;
    workPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

    __shared__ vid_t      firstFound[1024];

    vid_t     adj_offset = tx >> shifter;
    vid_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vid_t src = this_mp_start; src < this_mp_stop; src++) {
        auto vertex = hornet.vertex(src);
        vid_t srcLen = vertex.degree();

        for(int k = adj_offset; k < srcLen; k += number_blocks) {
            vid_t dest = vertex.edge(k).dst_id();
            degree_t destLen = hornet.vertex(dest).degree();
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

            /*
            if(large_len + small_len > cutoff)
                continue;
            */

            const vid_t* small_ptr = hornet.vertex(small).neighbor_ptr();
            const vid_t* large_ptr = hornet.vertex(large).neighbor_ptr();

            triangle_t triFound = count_triangles
                (hornet, small, small_ptr, small_len, large, large_ptr,
                 large_len, threads_per_block, (triangle_t*)firstFoundPos,
                 tx % threads_per_block, outPutTriangles,
                 nullptr, nullptr, 1, src, dest);

            atomicAdd(outPutTriangles+src,triFound);
            atomicAdd(outPutTriangles+dest,triFound);

        }
    }
}

void staticTriangleCounting(HornetGraph& hornet,
                        triangle_t* __restrict__ outPutTriangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        int cutoff,
                        HostDeviceVar<TriangleData> hd_data) {

    devicecuStaticTriangleCounting <<< thread_blocks, blockdim >>>
        (hornet.device_side(), outPutTriangles, threads_per_block,
         number_blocks, shifter, cutoff, hd_data);
    hd_data.sync();
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

TriangleCounting::TriangleCounting(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet),
                                       hd_triangleData(hornet){
}

TriangleCounting::~TriangleCounting(){
    release();
}

struct OPERATOR_InitTriangleCounts {
    HostDeviceVar<TriangleData> d_triangleData;

    OPERATOR (Vertex &vertex) {
        d_triangleData().triPerVertex[vertex.id()] = 0;
    }
};

void TriangleCounting::reset(){
    forAllVertices(hornet, OPERATOR_InitTriangleCounts { hd_triangleData });
}

void TriangleCounting::run(){
    run(0);
}

void TriangleCounting::run(int cutoff){

    staticTriangleCounting(hornet,
                        hd_triangleData().triPerVertex,
                        hd_triangleData().threadsPerIntersection,
                        hd_triangleData().numberInterPerBlock,
                        hd_triangleData().logThreadsPerInter,
                        hd_triangleData().threadBlocks,
                        hd_triangleData().blockSize,
                        cutoff,
                        hd_triangleData);
}

void TriangleCounting::release(){
    if(memReleased)
        return;
    memReleased=true;
    gpu::free(hd_triangleData().triPerVertex);

}


void TriangleCounting::setInitParameters(int threadBlocks, int blockSize, int threadsPerIntersection){
    hd_triangleData().threadBlocks                    = threadBlocks;
    hd_triangleData().blockSize                        = blockSize;

    if(hd_triangleData().blockSize%32 != 0){
        printf("The block size has to be a multiple of 32\n");
        printf("The block size has to be a reduced to the closet multiple of 32\n");
        hd_triangleData().blockSize = (hd_triangleData().blockSize/32)*32;
    }
    if(hd_triangleData().blockSize < 0){
        printf("The block size has to be a positive numbe\n");
        exit(0);
    }
    
    hd_triangleData().threadsPerIntersection = threadsPerIntersection;
    if(hd_triangleData().threadsPerIntersection <= 0 || hd_triangleData().threadsPerIntersection >32 ){
        printf("Threads per intersection have to be a power of two between 1 and 32\n");
        exit(0);
    }
    int temp = hd_triangleData().threadsPerIntersection,logtemp=0;
    while (temp>>=1) ++logtemp;
    hd_triangleData().logThreadsPerInter            = logtemp;
    hd_triangleData().numberInterPerBlock=hd_triangleData().blockSize/hd_triangleData().threadsPerIntersection;
}


void TriangleCounting::init(){
    memReleased=false;
    gpu::allocate(hd_triangleData().triPerVertex, hd_triangleData().nv+10);
    reset();
}

triangle_t TriangleCounting::countTriangles(){
    hd_triangleData.sync();
    triangle_t* outputArray = (triangle_t*)malloc((hd_triangleData().nv+2)*sizeof(triangle_t));
    gpu::copyToHost(hd_triangleData().triPerVertex, (hd_triangleData().nv+2), outputArray);
    triangle_t sum=0;
    for(int i=0; i<(hd_triangleData().nv); i++){
        // printf("%d %ld\n", i,outputArray[i]);
        sum+=outputArray[i];
    }
    free(outputArray);
    //triangle_t sum=gpu::reduce(hd_triangleData().triPerVertex, hd_triangleData().nv+1);

    return sum;
}
} // namespace hornets_nest

