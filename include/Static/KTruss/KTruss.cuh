#pragma once

#include "cuStingerAlg.hpp"

using triangle_t = int;

namespace custinger_alg {

struct KTrussData {
    KTrussData(const custinger::cuStinger& custinger) : activeQueue(custinger){}
    int maxK;

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int* isActive;
    int* offsetArray;
    int* trianglePerEdge;
    int* trianglePerVertex;

    vid_t* src;
    vid_t* dst;
    int    counter;
    int    activeVertices;

    TwoLevelQueue<vid_t> activeQueue; // Stores all the active vertices

    int fullTriangleIterations;

    vid_t nv;
    off_t ne;           // undirected-edges
    off_t ne_remaining; // undirected-edges
};

//==============================================================================

// Label propogation is based on the values from the previous iteration.
class KTruss : public StaticAlgorithm {
public:
    KTruss(cuStinger& custinger, custinger::BatchUpdate& batch_update);
    ~KTruss();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    //--------------------------------------------------------------------------
    void setInitParameters(vid_t nv, eoff_t ne, int tsp, int nbl, int shifter,
                           int blocks, int sps);
    void init();

    bool findTrussOfK(bool& stop);
    void      runForK(int maxK);

    void          runDynamic();
    bool findTrussOfKDynamic(bool& stop);
    void      runForKDynamic(int maxK);

    void   copyOffsetArrayHost(vid_t* hostOffsetArray);
    void copyOffsetArrayDevice(vid_t* deviceOffsetArray);
    void        resetEdgeArray();
    void      resetVertexArray();

    vid_t getIterationCount();
    vid_t           getMaxK();

private:
    KTrussData              hostKTrussData;
    KTrussData*             deviceKTrussData;
    custinger::BatchUpdate& batch_update;
};

//==============================================================================

void callDeviceDifferenceTriangles(const custinger::cuStinger& custinger,
                                   const custinger::BatchUpdate& batch_update,
                                   triangle_t* __restrict__ outPutTriangles,
                                   int threads_per_intersection,
                                   int num_intersec_perblock,
                                   int shifter,
                                   int thread_blocks,
                                   int blockdim,
                                   bool deletion);

namespace ktruss_operators {

__device__ __forceinline__
void init(const Vertex& vertex, void* metadata) {
    KTrussData*    kt = reinterpret_cast<KTrussData*>(metadata);
    vid_t         src = vertex.id();
    kt->isActive[src] = 1;
}

__device__ __forceinline__
void findUnderK(const Vertex& vertex, void* metadata) {
    KTrussData* kt = reinterpret_cast<KTrussData*>(metadata);
    //vid_t  src_len = custinger->dVD->used[src];
    vid_t  src_len = vertex.degree();
    vid_t      src = vertex.id();

    if (kt->isActive[src] == 0)
        return;
    if (src_len == 0) {
        kt->isActive[src] = 0;
        return;
    }
    //vid_t* adj_src = custinger->dVD->adj[src]->dst;
    for (vid_t adj = 0; adj < src_len; adj++) {
        //vid_t dst = adj_src[adj];
        auto edge = vertex.edge(adj);
        int   pos = kt->offsetArray[src] + adj;
        if (kt->trianglePerEdge[pos] < kt->maxK - 2) {
            int      spot = atomicAdd(&(kt->counter), 1);
            kt->src[spot] = src;
            //kt->dst[spot] = dst;
            kt->dst[spot] = edge.dst();
        }
    }
}

__device__ __forceinline__
void findUnderKDynamic(const Vertex& vertex, void* metadata) {
    KTrussData* kt = reinterpret_cast<KTrussData*>(metadata);
    //vid_t  src_len = custinger->dVD->used[src];
    vid_t src_len = vertex.degree();
    vid_t     src = vertex.id();

    if(kt->isActive[src] == 0)
        return;
    if(src_len == 0) {
        kt->isActive[src] = 0;
        return;
    }
    //vid_t* adj_src = custinger->dVD->adj[src]->dst;
    for (vid_t adj = 0; adj < src_len; adj++) {
        //vid_t dst = adj_src[adj];
        auto   edge = vertex.edge(adj);
        //if (custinger->dVD->adj[src]->ew[adj] < (kt->maxK - 2)) {
        if (edge.weight() < kt->maxK - 2) {
            int      spot = atomicAdd(&(kt->counter), 1);
            kt->src[spot] = src;
            //kt->dst[spot] = dst;
            kt->dst[spot] = edge.dst();
        }

        // if(src==111 && edge.dst()==322143){
        //     printf("%d %d %d\n", src,edge.dst(),edge.weight());
        // }
        // if(src==322143 && edge.dst()==111){
        //     printf("%d %d %d\n", src,edge.dst(),edge.weight());
        // }
    }
}

__device__ __forceinline__
 void queueActive(const Vertex& vertex, void* metadata) {
    KTrussData* kt = reinterpret_cast<KTrussData*>(metadata);
    //vid_t  src_len = custinger->dVD->used[src];
    vid_t  src_len = vertex.degree();
    vid_t      src = vertex.id();

    if (src_len == 0 && !kt->isActive[src])
        kt->isActive[src] = 0;
    else
        kt->activeQueue.insert(src);
}

__device__ __forceinline__
void countActive(const Vertex& vertex, void* metadata) {
    KTrussData* kt = reinterpret_cast<KTrussData*>(metadata);
    //vid_t  src_len = custinger->dVD->used[src];
    vid_t  src_len = vertex.degree();
    vid_t      src = vertex.id();

    if (src_len == 0 && !kt->isActive[src])
        kt->isActive[src] = 0;
    else
        atomicAdd(&(kt->activeVertices), 1);
}

__device__ __forceinline__
void resetWeights(const Vertex& vertex, void* metadata) {
    KTrussData* kt = reinterpret_cast<KTrussData*>(metadata);
    //vid_t  src_len = custinger->dVD->used[src];
    //int        pos = kt->offsetArray[src];
    vid_t  src_len = vertex.degree();
    int        pos = kt->offsetArray[vertex.id()];

    for (vid_t adj = 0; adj < src_len; adj++)
        //custinger->dVD->adj[src]->ew[adj] = kt->trianglePerEdge[pos + adj];
        vertex.edge(adj).set_weight(kt->trianglePerEdge[pos + adj]); //!!!
}

} // namespace ktruss_operators
} // namespace custinger_alg
