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


using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;
using clusterCoeff_t =  float;
//==============================================================================

class ClusteringCoefficient : public TriangleCounting2 {
public:
    ClusteringCoefficient(HornetGraph& hornet);
    ~ClusteringCoefficient();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void init();

    /// Array needs to be pre-allocated by user
    void copyLocalClusCoeffToHost(clusterCoeff_t* h_tcs);


private:
   clusterCoeff_t* d_ccLocal { nullptr };
};

//==============================================================================

} // namespace hornets_nest
