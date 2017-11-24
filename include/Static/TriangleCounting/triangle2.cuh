#pragma once

#include "HornetAlg.hpp"
#include "Core/HostDeviceVar.cuh"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

namespace hornets_nest {

//using triangle_t = int;
using triangle_t = unsigned long long int;
using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;


//==============================================================================

class TriangleCounting2 : public StaticAlgorithm<HornetGraph> {
public:
    TriangleCounting2(HornetGraph& hornet);
    ~TriangleCounting2();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void init();

    triangle_t countTriangles();

private:
   triangle_t* triPerVertex { nullptr };

};

//==============================================================================

} // namespace hornets_nest
