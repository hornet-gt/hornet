/**
 * @internal
 * @author Oded Green                                                  <br>
 *         Georgia Institute of Technology, Computational Science and Engineering                   <br>
 *         ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
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

#include <cuda_runtime.h>

#include "cuStingerAlg.hpp"
#include "Core/BatchUpdate.cuh"
#include "Core/StandardAPI.hpp"

#include "Static/KatzCentrality/katz.cuh"

using namespace std;
using namespace xlib;
using namespace custinger;


namespace custinger_alg {

class katzDynamicData: public katzData{
public:
	ulong_t*    newPathsCurr;
	ulong_t*    newPathsPrev;
	TwoLevelQueue<vid_t> activeQueue;
	// vertexQueue activeQueue; // Stores all the active vertices
	int*		active;
	int32_t iterationStatic;
};

using degree_t = vid_t;

class katzCentralityDynamic final:public StaticAlgorithm{
public:
    explicit katzCentralityDynamic(cuStinger& custinger);
    ~katzCentralityDynamic();


	void setInitParametersUndirected(int32_t maxIteration_, int32_t K_,degree_t maxDegree_);
	void setInitParametersDirected(int32_t maxIteration_, int32_t K_,degree_t maxDegree_,cuStinger* invertedGraph);

	void init();
	void reset() override;
	void run() override;
	void release() override;
    bool validate() override { return true; }

	void runStatic();

	void batchUpdateInserted(BatchUpdate &bu);
	void batchUpdateDeleted(BatchUpdate &bu);
	void Release();

	void SyncHostWithDevice(){
		gpu::copyDeviceToHost(deviceKatzData,1,&hostKatzData);
	}
	void SyncDeviceWithHost(){
		gpu::copyHostToDevice(&hostKatzData,1,deviceKatzData);
	}

	int32_t getIterationCount();

	virtual void copyKCToHost(double* hostArray){
		kcStatic.copyKCToHost(hostArray);
	}
	virtual void copynPathsToHost(ulong_t* hostArray){
		kcStatic.copynPathsToHost(hostArray);
	}
protected:
	katzDynamicData hostKatzData, *deviceKatzData;
private:
	void processUpdate(BatchUpdate &bu, bool isInsert);

    load_balacing::BinarySearch load_balacing;	
	// cusLoadBalance* cusLB;
	katzCentrality kcStatic;

	cuStinger* invertedGraph;
	bool isDirected;
	bool memReleased;
	
};

namespace katzCentralityDynamicOperator{

// Used only once when the streaming katz data structure is initialized
__device__ __forceinline__ 
void initStreaming(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	vid_t src = s.id();
	kd->newPathsCurr[src]=0;
	kd->newPathsPrev[src]= kd->nPaths[1][src];
	kd->active[src]=0;
}

__device__ __forceinline__
void setupInsertions(Vertex& src_, Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	auto dst = edge.dst_id();
    auto src = src_.id();
	atomicAdd(kd->KC+src, kd->alpha);
	atomicAdd(kd->newPathsPrev+src, 1);
	vid_t prev = atomicCAS(kd->active+src,0,kd->iteration);
	if(prev==0){
		kd->activeQueue.insert(src);
	}
}

__device__ __forceinline__
void setupDeletions(Vertex& src_, Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	double minusAlpha = -kd->alpha;
	auto dst = edge.dst_id();
    auto src = src_.id();

	atomicAdd(kd->KC+src, minusAlpha);
	atomicAdd(kd->newPathsPrev+src, -1);
	vid_t prev = atomicCAS(kd->active+src,0,kd->iteration);
	if(prev==0){
		kd->activeQueue.insert(src);
	}
}

__device__ __forceinline__
void initActiveNewPaths(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	vid_t src = s.id();
	kd->newPathsCurr[src]= kd->nPaths[kd->iteration][src];
}

__device__ __forceinline__
void findNextActive(Vertex& src_, Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	auto dst = edge.dst_id();
    auto src = src_.id();

	vid_t prev = atomicCAS(kd->active+dst,0,kd->iteration);
	if(prev==0){
		kd->activeQueue.insert(dst);
		kd->newPathsCurr[dst]= kd->nPaths[kd->iteration][dst];
	}
}

__device__ __forceinline__
void updateActiveNewPaths(Vertex& src_, Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	auto dst = edge.dst_id();
    auto src = src_.id();

	if(kd->active[src] < kd->iteration){
		ulong_t valToAdd = kd->newPathsPrev[src] - kd->nPaths[kd->iteration-1][src];
		atomicAdd(kd->newPathsCurr+dst, valToAdd);
	}
}

__device__ __forceinline__
void updateNewPathsBatchInsert(Vertex& src_, Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	auto dst = edge.dst_id();
    auto src = src_.id();

	ulong_t valToAdd = kd->nPaths[kd->iteration-1][dst];
	atomicAdd(kd->newPathsCurr+src, valToAdd);
}

__device__ __forceinline__
void updateNewPathsBatchDelete(Vertex& src_, Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	auto dst = edge.dst_id();
    auto src = src_.id();

	ulong_t valToRemove = -kd->nPaths[kd->iteration-1][dst];
	atomicAdd(kd->newPathsCurr+src, valToRemove);
}


__device__ __forceinline__
void updatePrevWithCurr(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	vid_t src = s.id();

	// Note the conversion to signed long long int!! Especially important for edge deletions where this diff can be negative
	long long int pathsDiff = kd->newPathsCurr[src] - kd->nPaths[kd->iteration][src];
	
	kd->KC[src] += kd->alphaI*(pathsDiff);
	if(kd->active[src] < kd->iteration){
		kd->nPaths[kd->iteration-1][src] = kd->newPathsPrev[src];
	}
	kd->newPathsPrev[src] = kd->newPathsCurr[src];
}

__device__ __forceinline__
void updateLastIteration(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzDynamicData*>(optional_field);
	vid_t src = s.id();

	if(kd->active[src] < (kd->iteration)){
		kd->nPaths[kd->iteration-1][src] = kd->newPathsPrev[src];
	}
}


};



} // cuStingerAlgs namespace
