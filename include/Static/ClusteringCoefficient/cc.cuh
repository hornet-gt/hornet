#pragma once

#include "HornetAlg.hpp"
#include "Core/HostDeviceVar.cuh"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

#include "Static/TriangleCounting/triangle2.cuh"


namespace hornets_nest {

//using triangle_t = int;
using triangle_t = unsigned long long;
using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;


//==============================================================================

class ClusteringCoefficient : public StaticAlgorithm<HornetGraph> {
public:
    ClusteringCoefficient(HornetGraph& hornet);
    ~ClusteringCoefficient();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void init();
    // void copyTCToHost(triangle_t* h_tcs);


private:
   float* ccLocal { nullptr };
   TriangleCounting2* tri;
};

//==============================================================================

} // namespace hornets_nest
