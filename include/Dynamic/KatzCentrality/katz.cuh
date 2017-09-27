/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
 *   ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "HornetAlg.hpp"
#include "Core/BatchUpdate.cuh"
#include "Core/StandardAPI.hpp"
#include "Static/KatzCentrality/katz.cuh"

namespace hornet_alg {

struct katzDynamicData : katzData {
    TwoLevelQueue<vid_t> active_queue;

    ulong_t* new_paths_curr;
    ulong_t* new_paths_prev;
    int*     active;
    int      iteration_static;
};

class katzCentralityDynamic : public StaticAlgorithm<HornetGPU> {
public:
    katzCentralityDynamic(HornetGPU& hornet);
    ~katzCentralityDynamic();

    void setInitParametersUndirected(int maxIteration_, int K_, degree_t maxDegree_);
    void setInitParametersDirected(int maxIteration_, int K_, degree_t maxDegree_,
                                   cuStinger* invertedGraph);

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    void runStatic();

    void batchUpdateInserted(BatchUpdate &bu);
    void batchUpdateDeleted(BatchUpdate &bu);
    void Release();

    int get_iteration_count();

    virtual void copyKCToHost(double* hostArray){
        kcStatic.copyKCToHost(hostArray);
    }
    virtual void copynPathsToHost(ulong_t* hostArray){
        kcStatic.copynPathsToHost(hostArray);
    }
//protected:
//    katzDynamicData hostKatzData, *deviceKatzData;
private:
    HostDeviceVar<KatzData> hd_katzdata;

    load_balacing::BinarySearch load_balacing;
    katzCentrality kcStatic;

    cuStinger* invertedGraph;
    bool isDirected;

    void processUpdate(BatchUpdate &bu, bool isInsert);
};

} // namespace hornet_alg
