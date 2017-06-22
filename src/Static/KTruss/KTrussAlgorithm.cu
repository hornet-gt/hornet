#include "update.hpp"
#include "cuStinger.hpp"
#include "operators.cuh"
#include "static_k_truss/k_truss.cuh"

namespace custinger_alg {

__device__ __forceinline__
void initialize(vid_t diag_id,
                vid_t u_len,
                vid_t v_len,
                const vid_t* __restrict__ u_min,
                const vid_t* __restrict__ u_max,
                const vid_t* __restrict__ v_min,
                const vid_t* __restrict__ v_max,
                const int*   __restrict__ found) {
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
void workPerThread(vid_t uLength,
                   vid_t vLength,
                   int threadsPerIntersection,
                   int threadId,
                   const int* __restrict__ outWorkPerThread,
                   const int* __restrict__ outDiagonalId) {
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
             vid_t    diagonalId,
             const vid_t*  __restrict__ uNodes,
             const vid_t*  __restrict__ vNodes,
             const vid_t*  __restrict__ uLength,
             const vid_t* __restrict__ outUMin,
             const vid_t* __restrict__ outUMax,
             const vid_t* __restrict__ outVMin,
             const vid_t* __restrict__ outVMax,
             const vid_t* __restrict__ outUCurr,
             const vid_t* __restrict__ outVCurr) {
    vid_t length;

    while (!found) {
        *outUCurr = (*outUMin + *outUMax) / 2u;
        *outVCurr = diagonalId - *outUCurr;
        if (*outVCurr >= *outVMax) {
            length = *outUMax - *outUMin;
            if (length == 1) {
                found = 1;
                continue;
            }
        }

        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (comp1 && !comp2)
            found = 1;
        else if (comp1) {
          *outVMin = *outVCurr;
          *outUMax = *outUCurr;
        }
        else {
          *outVMax = *outVCurr;
          *outUMin = *outUCurr;
        }
      }

    if (*outVCurr >= *outVMax && length == 1 && *outVCurr > 0 &&
        *outUCurr > 0 && *outUCurr < *uLength - 1) {
        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (!comp1 && !comp2) {
            (*outUCurr)++;
            (*outVCurr)--;
        }
    }
}

__device__ __forceinline__
int fixStartPoint(vid_t uLength, vid_t vLength,
                  const vid_t* __restrict__ uCurr,
                  const vid_t* __restrict__ vCurr,
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

/*
__device__ __forceinline__
vid_t* binSearch(vid_t *a, vertexId_t x, vid_t n) {
    vid_t min = 0, max = n, acurr, curr;// = (min+max)/2
    do {
        curr  = (min + max) / 2;
        acurr = a[curr];
        min   = (x > acurr) ? curr : min;
        max   = (x < acurr) ? curr : max;
    } while (x != acurr || min != max);
    return a + curr;
}*/

/*
__device__ __forceinline__
int findIndexOfVertex(cuStinger* custinger, vid_t src, vid_t dst__) {
    vid_t   srcLen = custinger->dVD->used[src];
    vid_t* adj_src = custinger->dVD->adj[src]->dst;

    for (vid_t adj = 0; adj < srcLen; adj++) {
        vid_t dst = adj_src[adj];
        if (dst == dst__)
            return adj;
    }
#if !defined(NDEBUG)
    printf("This should never happpen\n");
#endif
    return -1;
}*/

__device__ __forceinline__
void indexBinarySearch(vid_t* data, vid_t arrLen, vid_t key, int& pos) {
    int  low = 0;
    int high = arrLen - 1;
    while (high >= low) {
         int middle = (low + high) / 2;
         if (data[middle] == key) {
             pos = middle;
             return;
         }
         if(data[middle] < key)
             low = middle + 1;
         if(data[middle] > key)
             high = middle - 1;
    }
}

__device__ __forceinline__
void findIndexOfTwoVerticesBinary(const Vertex& vertex,
                                  vid_t src, vid_t v1, vid_t v2,
                                  int &pos_v1, int &pos_v2) {
    //vid_t* adj_src = custinger->dVD->adj[src]->dst;
    //vid_t   srcLen = custinger->dVD->used[src];
    vid_t   srcLen = vertex.degree();
    vid_t* adj_src = vertex.edge_ptr();

    pos_v1 = -1;
    pos_v2 = -1;

    indexBinarySearch(adj_src, srcLen, v1, pos_v1);
    indexBinarySearch(adj_src, srcLen, v2, pos_v2);
}

__device__ __forceinline__
void findIndexOfTwoVertices(const Vertex& vertex, vid_t v1, vid_t v2,
                            int &pos_v1, int &pos_v2) {
    //vid_t   srcLen = custinger->dVD->used[src];
    //vid_t* adj_src = custinger->dVD->adj[src]->dst;
    vid_t   srcLen = vertex.degree();
    vid_t* adj_src = vertex.edge_ptr();

    pos_v1 = -1;
    pos_v2 = -1;
    for(vid_t adj = 0; adj < srcLen; adj += 1) {
        vid_t dst = adj_src[adj];
        if (dst == v1)
            pos_v1 = adj;
        if (dst == v2)
            pos_v2 = adj;
        if (pos_v1 != -1 && pos_v2 != -1)
            return;
    }
#if !defined(NDEBUG)
    printf("This should never happpen\n");
#endif
}

template<bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ __forceinline__
void intersectCount(const cuStingerDevData& custinger,
                    vid_t uLength, vid_t vLength,
                    const vid_t*  __restrict__ uNodes,
                    const vid_t*  __restrict__ vNodes,
                    const vid_t*  __restrict__ uCurr,
                    const vid_t*  __restrict__ vCurr,
                    const int*    __restrict__ workIndex,
                    const int*    __restrict__ workPerThread,
                    const int*    __restrict__ triangles,
                    int found,
                    const triangle_t*  __restrict__ outPutTriangles,
                    const vid_t*  __restrict__ uMask,
                    const vid_t*  __restrict__ vMask,
                    triangle_t multiplier,
                    vid_t src, vid_t dest,
                    vid_t u, vid_t v) {

    if (*uCurr < uLength && *vCurr < vLength) {
        int comp;
        int vmask;
        int umask;
        while (*workIndex < *workPerThread) {
            vmask = vMasked ? vMask[*vCurr] : 0;
            umask = uMasked ? uMask[*uCurr] : 0;
            comp  = uNodes[*uCurr] - vNodes[*vCurr];

            *triangles += (comp == 0 && !umask && !vmask);

            if (upd3rdV && comp == 0 && !umask && !vmask) {
                if (subtract) {
                    // atomicSub(outPutTriangles + uNodes[*uCurr], multiplier);

                    // Ktruss
                    vid_t common = uNodes[*uCurr];
                    length_t pos_id;

                    Vertex vertex_common(custinger, common);
                    auto edge_weight_ptr = vertex_common.edge_weight_ptr();
                    vid_t posu, posv;
                    //findIndexOfTwoVerticesBinary(custinger, common, u, v,
                    //                             posu, posv);
                    findIndexOfTwoVerticesBinary(vertex_common, u, v,
                                                 posu, posv);

                    if (posu != -1)
                        atomicSub(edge_weight_ptr + posu, 1);
                        //atomicSub(custinger->dVD->adj[common]->ew + posu, 1);
#if !defined(NDEBUG)
                    else
                        printf("1");
#endif
                    if (posv != -1)
                        atomicSub(edge_weight_ptr + posv, 1);
                        //atomicSub(custinger->dVD->adj[common]->ew + posv, 1);
#if !defined(NDEBUG)
                    else
                        printf("2");
#endif
                    Vertex vertex_u(custinger, u);
                    Vertex vertex_v(custinger, v);
                    //atomicSub(custinger->dVD->adj[u]->ew + *uCurr, 1);
                    //atomicSub(custinger->dVD->adj[v]->ew + *vCurr, 1);
                    atomicSub(vertex_u.edge_weight_ptr() + *uCurr, 1);
                    atomicSub(vertex_v.edge_weight_ptr() + *vCurr, 1);
                }
            }

            *uCurr     += (comp <= 0 && !vmask) || umask;
            *vCurr     += (comp >= 0 && !umask) || vmask;
            *workIndex += (comp == 0&& !umask && !vmask) + 1;

            if (*vCurr == vLength || *uCurr == uLength)
                break;
        }
        *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
    }
}

// u_len < v_len
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ __forceinline__
triangle_t count_triangles(const cuStingerDevData& custinger
                           vid_t u,
                           const vid_t* __restrict__ u_nodes,
                           vid_t u_len,
                           vid_t v,
                           const vid_t* __restrict__ v_nodes,
                           vid_t v_len,
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
        intersectCount<uMasked, vMasked, subtract, upd3rdV>
            (custinger, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
            &work_index, &work_per_thread, &triangles, firstFound[tId],
            outPutTriangles, uMask, vMask, multiplier, src, dest, u, v);
    }
    return triangles;
}

__device__ __forceinline__
void workPerBlock(vid_t numVertices,
                  const vid_t* __restrict__ outMpStart,
                  const vid_t* __restrict__ outMpEnd,
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

//==============================================================================

__global__
void devicecuStingerKTruss(cuStingerDevData custinger,
                           const triangle_t* __restrict__ outPutTriangles,
                           int threads_per_block,
                           int number_blocks,
                           int shifter,
                           kTrussData* __restrict__ devData) {
    vid_t nv = custinger->nv;
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements
    //to intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vid_t this_mp_start, this_mp_stop;

    const int blockSize = blockDim.x;
    workPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

    __shared__ triangle_t s_triangles[1024];
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
            //auto edge = vertex.edge(k);
            vid_t dest = vertex.edge(k).dst();
            //int destLen = custinger->dVD->getUsed()[dest];
            int destLen = Vertex(custinger, dest).degree();

            if (dest<src)
                continue;

            bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
            if (avoidCalc)
                continue;

            bool sourceSmaller = srcLen < destLen;
            vid_t        small = sourceSmaller ? src : dest;
            vid_t        large = sourceSmaller ? dest : src;
            vid_t    small_len = sourceSmaller ? srcLen : destLen;
            vid_t    large_len = sourceSmaller ? destLen : srcLen;

            //const vid_t* small_ptr = custinger->dVD->getAdj()[small]->dst;
            //const vid_t* large_ptr = custinger->dVD->getAdj()[large]->dst;
            const vid_t* small_ptr = Vertex(custinger, small).edge_ptr();
            const vid_t* large_ptr = Vertex(custinger, large).edge_ptr();

            // triangle_t triFound = count_triangles<false,false,false,true>
            triangle_t triFound = count_triangles<false, false, false, false>
                (custinger, small, small_ptr, small_len, large, large_ptr,
                 large_len, threads_per_block, firstFoundPos,
                 tx % threads_per_block, outPutTriangles,
                 nullptr, nullptr, 1, src, dest);
            tCount += triFound;
            int pos = devData->offsetArray[src] + k;
            atomicAdd(devData->trianglePerEdge + pos, triFound);
            pos = -1;
            //indexBinarySearch(custinger->dVD->getAdj()[dest]->dst
            //                  destLen, src,pos);
            indexBinarySearch(Vertex(custinger, dest).edge_ptr(),
                              destLen, src,pos);

            pos = devData->offsetArray[dest] + pos;
            atomicAdd(devData->trianglePerEdge + pos, triFound);
        }
    //    s_triangles[tx] = tCount;
    //    blockReduce(&outPutTriangles[src],s_triangles,blockSize);
    }
}

void kTrussOneIteration(cuStinger& custinger,
                        const triangle_t* __restrict__ outPutTriangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        kTrussData* __restrict__ devData) {

    //devicecuStingerKTruss <<< thread_blocks, blockdim >>>
    //    (custinger.devicePtr(), outPutTriangles, threads_per_block,
    //     number_blocks, shifter, devData);
    devicecuStingerKTruss <<< thread_blocks, blockdim >>>
        (custinger.device_data(), outPutTriangles, threads_per_block,
         number_blocks, shifter, devData);
}

__global__
void devicecuStingerNewTriangles(cuStingerDevData custinger,
                                 BatchUpdateData*  __restrict__ bud,
                                 const triangle_t* __restrict__ outPutTriangles,
                                 int threads_per_block,
                                 int number_blocks,
                                 int shifter,
                                 bool deletion,
                                 const vid_t* __restrict__ redCU) {
    vid_t batchSize = *(bud->getBatchSize());
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vid_t this_mp_start, this_mp_stop;

    // length_t *d_off = bud->getOffsets();
    vid_t* d_ind = bud->getDst();
    vid_t* d_seg = bud->getSrc();

    workPerBlock(batchSize, &this_mp_start, &this_mp_stop, blockDim.x);

    __shared__ vid_t firstFound[1024];

    vid_t     adj_offset = tx >> shifter;
    vid_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vid_t edge = this_mp_start + adj_offset; edge < this_mp_stop;
         edge += number_blocks){
        if (bud->getIndDuplicate()[edge] == 1) // this means it's a duplicate edge
            continue;

        vid_t src  = d_seg[edge];
        vid_t dest = d_ind[edge];

        if (src < dest)
            continue;

        vid_t srcLen  = Vertex(custinger, src).degree();
        vid_t destLen = Vertex(custinger, dest).degree();
        //vid_t srcLen  = custinger->dVD->getUsed()[src];
        //vid_t destLen = custinger->dVD->getUsed()[dest];

        bool avoidCalc = (src == dest) || (destLen == 0) || (srcLen == 0);
        if (avoidCalc)
            continue;

        bool sourceSmaller = srcLen < destLen;
        vid_t        small = sourceSmaller ? src : dest;
        vid_t        large = sourceSmaller ? dest : src;
        vid_t    small_len = sourceSmaller ? srcLen : destLen;
        vid_t    large_len = sourceSmaller ? destLen : srcLen;
#if !defined(NDEBUG)
        if (small_len == 0)
            printf("hello oded\n");
#endif
        //const vid_t* small_ptr = custinger->dVD->getAdj()[small]->dst;
        //const vid_t* large_ptr = custinger->dVD->getAdj()[large]->dst;
        const vid_t* small_ptr = Vertex(custinger, small).edge_ptr();
        const vid_t* large_ptr = Vertex(custinger, large).edge_ptr();

        triangle_t tCount = count_triangles<false, false, true, true>(
                                custinger, small, small_ptr, small_len,
                                large,large_ptr, large_len,
                                threads_per_block, firstFoundPos,
                                tx % threads_per_block, outPutTriangles,
                                nullptr, nullptr, 2, src, dest);
        __syncthreads();
    }
}

//==============================================================================

template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ __forceinline__
void intersectCountAsymmetric(const cuStingerDevData& custinger,
                              vid_t uLength, vid_t vLength,
                              const vid_t* __restrict__ uNodes,
                              const vid_t* __restrict__ vNodes,
                              const vid_t* __restrict__ uCurr,
                              const vid_t* __restrict__ vCurr,
                              const int*   __restrict__ workIndex,
                              const int*   __restrict__ workPerThread,
                              const int*   __restrict__ triangles,
                              int found,
                              const triangle_t* __restrict__ outPutTriangles,
                              const vid_t*      __restrict__ uMask,
                              const vid_t*      __restrict__ vMask,
                              triangle_t multiplier,
                              vid_t src, vid_t dest,
                              vid_t u, vid_t v) {

    if (*uCurr < uLength && *vCurr < vLength) {
        int comp, vmask, umask;
        while (*workIndex < *workPerThread) {
            vmask = vMasked ? vMask[*vCurr] : 0;
            umask = uMasked ? uMask[*uCurr] : 0;
            comp  = uNodes[*uCurr] - vNodes[*vCurr];

            *triangles += (comp == 0 && !umask && !vmask);

            if (upd3rdV && comp == 0 && !umask && !vmask) {
                if (subtract) {
                    atomicSub(outPutTriangles + uNodes[*uCurr], multiplier);

                    // Ktruss
                    vid_t common = uNodes[*uCurr];

                    if(dest==u) {
                        auto w_ptr = vertex(custinger, dest).edge_weight_ptr();
                        atomicSub(w_ptr, V + *uCurr, 1);
                        //atomicSub(custinger->dVD->adj[dest]->ew + *uCurr, 1);
                    }
                    else {
                        auto w_ptr = vertex(custinger, dest).edge_weight_ptr();
                        atomicSub(w_ptr + *vCurr, 1);
                        //atomicSub(custinger->dVD->adj[dest]->ew + *vCurr, 1);
                    }
                }
            }
            *uCurr     += (comp <= 0 && !vmask) || umask;
            *vCurr     += (comp >= 0 && !umask) || vmask;
            *workIndex += (comp == 0 && !umask && !vmask) + 1;

            if (*vCurr == vLength || *uCurr == uLength)
                break;
        }
        *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
    }
}

// u_len < v_len
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV>
__device__ __forceinline__
triangle_t count_trianglesAsymmetric(
                                 const cuStingerDevData& custinger,
                                 vid_t u,
                                 const vid_t* __restrict__ u_nodes,
                                 vid_t u_len,
                                 vid_t v,
                                 const vid_t* __restrict__ v_nodes,
                                 vid_t v_len,
                                 int threads_per_block,
                                 volatile vid_t* __restrict__ firstFound,
                                 int tId,
                                 const triangle_t* __restrict__ outPutTriangles,
                                 const vid_t* __restrict__ uMask,
                                 const vid_t* __restrict__ vMask,
                                 triangle_t multiplier,
                                 vid_t src, vid_t dest) {
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    // Tersect - this number will be off by 1.
    int work_per_thread, diag_id;
    workPerThread(u_len, v_len, threads_per_block, tId,
                  &work_per_thread, &diag_id);
    triangle_t triangles = 0;
    int       work_index = 0,
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
            firstFound[tId-1] = sum;
        triangles += sum;
        intersectCountAsymmetric<uMasked, vMasked, subtract, upd3rdV>
            (custinger, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
             &work_index, &work_per_thread, &triangles, firstFound[tId],
             outPutTriangles, uMask, vMask, multiplier, src, dest, u, v);
    }
    return triangles;
}

//==============================================================================

__global__
void deviceBUTwoCUOneTriangles(cuStingerDevData custinger,
                                BatchUpdateData *bud,
                                const triangle_t* __restrict__ outPutTriangles,
                                int  threads_per_block,
                                int  number_blocks,
                                int  shifter,
                                bool deletion,
                                const vid_t* __restrict__ redCU,
                                const vid_t* __restrict__ redBU) {
    vid_t batchsize = *(bud->getBatchSize());
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vid_t this_mp_start, this_mp_stop;

    vid_t* d_off = bud->getOffsets();
    vid_t* d_ind = bud->getDst();
    vid_t* d_seg = bud->getSrc();

    int blockSize = blockDim.x;
    workPerBlock(batchsize, &this_mp_start, &this_mp_stop, blockSize);

    __shared__ vid_t firstFound[1024];

    vid_t     adj_offset = tx >> shifter;
    vid_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vid_t edge = this_mp_start+adj_offset; edge < this_mp_stop;
            edge+=number_blocks) {
        if (bud->getIndDuplicate()[edge]) // this means it's a duplicate edge
            continue;

        vid_t src = bud->getSrc()[edge];
        vid_t dest= bud->getDst()[edge];

        // vid_t srcLen= redBU[src];
        // vid_t destLen=redCU[dest];
        vid_t  srcLen = d_off[src + 1] - d_off[src];
        vid_t destLen = custinger->dVD->getUsed()[dest];

        bool avoidCalc = (src == dest) || (srcLen == 0);
        if (avoidCalc)
            continue;

        const vid_t*      src_ptr = d_ind + d_off[src];
        const vid_t* src_mask_ptr = bud->getIndDuplicate() + d_off[src];
        //const vid_t*      dst_ptr = custinger->dVD->getAdj()[dest]->dst;
        const vid_t*      dst_ptr = Vertex(custinger, dest).edge_ptr();

        bool sourceSmaller = srcLen < destLen;
        vid_t        small = sourceSmaller ? src : dest;
        vid_t        large = sourceSmaller ? dest : src;
        vid_t    small_len = sourceSmaller ? srcLen : destLen;
        vid_t    large_len = sourceSmaller ? destLen : srcLen;

        const vid_t*      small_ptr = sourceSmaller? src_ptr : dst_ptr;
        const vid_t* small_mask_ptr = sourceSmaller? src_mask_ptr : nullptr;
        const vid_t*      large_ptr = sourceSmaller? dst_ptr : src_ptr;
        const vid_t* large_mask_ptr = sourceSmaller? nullptr : src_mask_ptr;

        // triangle_t tCount=0;
        triangle_t tCount = sourceSmaller ?
                            count_trianglesAsymmetric<true,false,true,true>
                                (custinger, small, small_ptr, small_len,
                                  large, large_ptr, large_len,
                                 threads_per_block,firstFoundPos,
                                 tx % threads_per_block, outPutTriangles,
                                   small_mask_ptr, large_mask_ptr, 1,src,dest) :
                            count_trianglesAsymmetric<false,true,true,true>
                                (custinger, small, small_ptr, small_len,
                                 large, large_ptr, large_len,
                                   threads_per_block, firstFoundPos,
                                 tx % threads_per_block, outPutTriangles,
                                 small_mask_ptr, large_mask_ptr, 1,src,dest);

        atomicSub(outPutTriangles + src, tCount * 1);
        atomicSub(outPutTriangles + dest, tCount * 1);
        __syncthreads();
    }
}

void callDeviceDifferenceTriangles(
                                const cuStingerDevData& custinger,
                                const BatchUpdate& batch_update,
                                const triangle_t* __restrict__ outPutTriangles,
                                int  threads_per_intersection,
                                int  num_intersec_perblock,
                                int  shifter,
                                int  thread_blocks,
                                int  blockdim,
                                bool deletion) {
    dim3 numBlocks(1, 1);
    //vid_t batchsize = *(batch_update.getHostBUD()->getBatchSize());
    //vid_t        nv = *(batch_update.getHostBUD()->getNumVertices());
    vid_t batchsize = batch_update.size();

    vid_t        nv = *(batch_update.getHostBUD()->getNumVertices());
    ///???

    numBlocks.x = ceil( (float) nv / (float) blockdim );
    vid_t* redCU;
    vid_t* redBU;

    numBlocks.x = ceil( (float) (batchsize * threads_per_intersection) /
                        (float) blockdim );

    // cout << "The block dim is " << blockdim << " and the number of blocks is"
    //<< numBlocks.x << endl;
    // Calculate all new traingles regardless of repetition
    devicecuStingerNewTriangles <<< numBlocks, blockdim >>>
        (custinger.device_data(), batch_update.device_ptr(),
         outPutTriangles, threads_per_intersection, num_intersec_perblock,
         shifter,deletion, redCU);

    // Calculate triangles formed by ALL new edges
        // deviceBUThreeTriangles<<<numBlocks,blockdim>>>(custinger.devicePtr(),
    //    batch_update.getDeviceBUD()->devicePtr(), outPutTriangles,
    //threads_per_intersection,num_intersec_perblock,shifter,deletion,redBU);

    // Calculate triangles formed by two new edges
    deviceBUTwoCUOneTriangles <<< numBlocks, blockdim >>>
        (custinger.device_data(), batch_update.device_ptr(),
        outPutTriangles, threads_per_intersection, num_intersec_perblock,
        shifter, deletion, redCU, redBU);
}

} // namespace custinger_alg
