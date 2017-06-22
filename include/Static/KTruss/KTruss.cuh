#pragma once

#include "cuStingerAlg.hpp"

using triangle_t = int;

namespace custinger_alg {

struct KTrussData {
    int maxK;

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int* isActive;
    int* offsetArray;
    int* trianglePerEdge;
    int*
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
    explicit KTruss(cuStinger& custinger);
    ~KTruss();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    //--------------------------------------------------------------------------
    void setInitParameters(vid_t nv, off_t ne, int tsp, int nbl, int shifter,
                           int blocks, int sps);
    void init();

    bool findTrussOfK(cuStinger& custinger,bool& stop);
    void      runForK(cuStinger& custinger,int maxK);

    void          runDynamic(cuStinger& custinger);
    bool findTrussOfKDynamic(cuStinger& custinger, bool& stop);
    void      runForKDynamic(cuStinger& custinger, int maxK);

    void   copyOffsetArrayHost(vid_t* hostOffsetArray);
    void copyOffsetArrayDevice(vid_t* deviceOffsetArray);
    void        resetEdgeArray();
    void      resetVertexArray();

    vid_t getIterationCount();
    vid_t           getMaxK();

private:
    KTrussData hostKTrussData;
    //KTrussData* deviceKTrussData;
};

//==============================================================================

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
    vid_t  src_len = vertex.degree();
    int        pos = kt->offsetArray[src];

    for (vid_t adj = 0; adj < src_len; adj++)
        //custinger->dVD->adj[src]->ew[adj] = kt->trianglePerEdge[pos + adj];
        vertex.edge(adj).set_weight(kt->trianglePerEdge[pos + adj]);
}

} // namespace ktruss_operators
} // namespace custinger_alg
