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

using ulong_t = unsigned long long int;

using namespace std;

namespace custinger_alg {

struct katzData{
	ulong_t*   nPathsData;
	ulong_t**  nPaths;     // Will be used for dynamic graph algorithm which requires storing paths of all iterations.

	ulong_t*   nPathsCurr;
	ulong_t*   nPathsPrev;

	double*     KC;
	double*     lowerBound;
	double*     upperBound;

	double alpha;
	double alphaI; // Alpha to the power of I  (being the iteration)

	double lowerBoundConst;
	double upperBoundConst;

	int32_t K;

	int32_t maxDegree;
	int32_t iteration;
	int32_t maxIteration;
	// number of active vertices at each iteration
	int32_t nActive;
	int32_t nv;

	bool* isActive;
	double*     lowerBoundUnsorted;
	double*     lowerBoundSorted;
	int32_t*    vertexArrayUnsorted; // Sorting
	int32_t*    vertexArraySorted; // Sorting

};

// Label propogation is based on the values from the previous iteration.
class katzCentrality final:public StaticAlgorithm{
public:
    explicit katzCentrality(cuStinger& custinger);
    ~katzCentrality();

	void setInitParameters(int32_t maxIteration_,int32_t K_,int32_t maxDegree_, bool isStatic_=true);
	void init();
	void reset() override;
	void run() override;
	void release() override;
    bool validate() override { return true; }

	int32_t getIterationCount();

	const katzData getHostKatzData(){return hostKatzData;}
	const katzData* getDeviceKatzData(){return deviceKatzData;}

	virtual void copyKCToHost(double* hostArray){
		cout << "This is a stub" << endl;
		// copyArrayDeviceToHost(hostKatzData->KC,hostArray, hostKatzData->nv, sizeof(double));
	}

	virtual void copynPathsToHost(ulong_t* hostArray){
		cout << "This is a stub" << endl;
		// copyArrayDeviceToHost(hostKatzData->nPathsData,hostArray, (hostKatzData->nv)*hostKatzData->maxIteration, sizeof(ulong_t));
	}


protected:
	// katzData hostKatzData, *deviceKatzData;
	katzData hostKatzData, *deviceKatzData;

private:
    load_balacing::BinarySearch load_balacing;
	// cusLoadBalance* cusLB;
	bool isStatic;
	ulong_t** hPathsPtr;  // Will be used to store pointers to all iterations of the Katz centrality results

	bool memReleased;
};


namespace katz_operators {


// Used at the very beginning
__device__ __forceinline__
void init(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
	vid_t src = s.id();
	kd->nPathsPrev[src]=1;
	kd->nPathsCurr[src]=0;
	kd->KC[src]=0.0;
	kd->isActive[src]=true;
	// kd->vertexArrayUnsorted[src]=src;
}

__device__ __forceinline__
void initNumPathsPerIteration(Vertex& src, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
	kd->nPathsCurr[src.id()]=0;
}

__device__ __forceinline__
void updatePathCount(Vertex& src_, Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
	auto dst = edge.dst_id();
    auto src = src_.id();
	atomicAdd(kd->nPathsCurr+src, kd->nPathsPrev[dst]);
}


__device__ __forceinline__
void updateKatzAndBounds(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
    vid_t src = s.id();

	kd->KC[src]=kd->KC[src] + kd->alphaI * (double)kd->nPathsCurr[src];
	kd->lowerBound[src]=kd->KC[src] + kd->lowerBoundConst * (double)kd->nPathsCurr[src];
	kd->upperBound[src]=kd->KC[src] + kd->upperBoundConst * (double)kd->nPathsCurr[src];

	if(kd->isActive[src]){
		int32_t pos = atomicAdd(&(kd -> nActive),1);
		kd->vertexArrayUnsorted[pos] = src;
		kd->lowerBoundUnsorted[pos]=kd->lowerBound[src];
	}
}


__device__ __forceinline__
void countActive(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
	vid_t src = s.id();
	if (kd->upperBound[src] > kd->lowerBound[kd->vertexArraySorted[kd->K-1]]) {
		atomicAdd(&(kd -> nActive),1);
	}
	else{
		kd->isActive[src] = false;
	}
}

__device__ __forceinline__
void printPointers(cuStinger* custing,int32_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	if(threadIdx.x==0 && blockIdx.x==0 && src==0)
		printf("\n@ %p %p %p %p %p %p %p %p @\n",kd->nPathsData,kd->nPaths, kd->nPathsPrev, kd->nPathsCurr, kd->KC,kd->lowerBound,kd->lowerBoundUnsorted,kd->upperBound);
}

__device__ __forceinline__
void printKID(cuStinger* custing,int32_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	if(kd->nPathsPrev[src]!=1)
		printf("%d %ld\n ", src,kd->nPathsPrev[src]);
	if(kd->nPathsCurr[src]!=0)
		printf("%d %ld\n ", src,kd->nPathsCurr[src]);
}


};



} // cuStingerAlgs namespace
