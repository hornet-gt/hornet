#pragma once

#include "cuStingerAlg.hpp"

using triangle_t = int;

namespace custinger_alg {

struct TriangleData {
    TriangleData(const custinger::cuStinger& custinger){}

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int* trianglePerVertex;

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

    //--------------------------------------------------------------------------
    void setInitParameters(vid_t nv, eoff_t ne, int tsp, int nbl, int shifter,
                           int blocks, int sps);
    void init();

private:
    TriangleData   hostTriangleData;
    TriangleData*  deviceTriangleData;
};

//==============================================================================


namespace triangle_operators {
} // namespace ktruss_operators
} // namespace custinger_alg


