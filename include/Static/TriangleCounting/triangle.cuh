#pragma once

#include "cuStingerAlg.hpp"


using namespace custinger;

namespace custinger_alg {

using triangle_t = unsigned int;


struct TriangleData {
    TriangleData(const custinger::cuStinger& custinger){
        nv = custinger.nV();
        ne = custinger.nE();
        triPerVertex=NULL;
    }

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int threadBlocks;
    int blockSize;
    int threadsPerIntersection;
    int logThreadsPerInter;
    int numberInterPerBlock;

    triangle_t* triPerVertex;

    vid_t nv;
    off_t ne;           // undirected-edges
};

//==============================================================================

// Label propogation is based on the values from the previous iteration.
class TriangleCounting : public StaticAlgorithm {
public:
    TriangleCounting(cuStinger& custinger);
    ~TriangleCounting();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void init();
    void setInitParameters(int threadBlocks, int blockSize, int threadsPerIntersection);
    triangle_t countTriangles();

private:
    bool memReleased;
    TriangleData   hostTriangleData;
    TriangleData*  deviceTriangleData;
};

//==============================================================================


namespace triangle_operators {
    __device__ __forceinline__
    void init(vid_t src, void* optional_field) {
        auto tri = reinterpret_cast<TriangleData*>(optional_field);
        tri->triPerVertex[src] = 0;
    }
    
} // namespace triangle_operators
} // namespace custinger_alg


