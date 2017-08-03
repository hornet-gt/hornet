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
	double*     	lowerBoundSort;
	int32_t*     vertexArray; // Sorting
	int32_t*     indexArray; // Sorting

};

// Label propogation is based on the values from the previous iteration.
class katzCentrality final:public StaticAlgorithm{
public:
    explicit katzCentrality(cuStinger& custinger);
    ~katzCentrality();

	void setInitParameters(int32_t maxIteration_,int32_t K_,int32_t maxDegree_, bool isStatic_=true);
	void init(cuStinger& custing);
	void reset();
	// void run(cuStinger& custing);
	void run() override;
	void release();

	// virtual void SyncHostWithDevice(){
	// 	cout << "This is a stub" << endl;
	// 	// copyArrayDeviceToHost(deviceKatzData,hostKatzData,1, sizeof(katzData));
	// }
	// virtual void SyncDeviceWithHost(){
	// 	cout << "This is a stub" << endl;
	// 	// copyArrayHostToDevice(hostKatzData,deviceKatzData,1, sizeof(katzData));
	// }

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
// class katzCentralityOperator{
// public:

// Used at the very beginning
__device__ __forceinline__
void init(Vertex& s, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
	// katzData* kd = (katzData*)metadata;
	vid_t src = s.id();
	kd->nPathsPrev[src]=1;
	kd->nPathsCurr[src]=0;
	kd->KC[src]=0.0;
	kd->isActive[src]=true;
	kd->indexArray[src]=src;
}

// Used every iteration
__device__ __forceinline__
// void initNumPathsPerIteration(cuStinger* custing,int32_t src, void* metadata){
void initNumPathsPerIteration(Vertex& src, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
	// katzData* kd = (katzData*)metadata;
	kd->nPathsCurr[src.id()]=0;
}

// __device__ __forceinline__
// void updatePathCount(cuStinger* custing,int32_t src, int32_t dst, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	atomicAdd(kd->nPathsCurr+src, kd->nPathsPrev[dst]);
// }
__device__ __forceinline__
void updatePathCount(Vertex& src_, const Edge& edge, void* optional_field){
	auto kd = reinterpret_cast<katzData*>(optional_field);
	auto dst = edge.dst_id();
    auto src = src_.id();
	atomicAdd(kd->nPathsCurr+src, kd->nPathsPrev[dst]);
}


__device__ __forceinline__
void updateKatzAndBounds(cuStinger* custing,int32_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	kd->KC[src]=kd->KC[src] + kd->alphaI * (double)kd->nPathsCurr[src];
	kd->lowerBound[src]=kd->KC[src] + kd->lowerBoundConst * (double)kd->nPathsCurr[src];
	kd->upperBound[src]=kd->KC[src] + kd->upperBoundConst * (double)kd->nPathsCurr[src];   

	if(kd->isActive[src]){
		int32_t pos = atomicAdd(&(kd -> nActive),1);
		kd->vertexArray[pos] = src;
		kd->lowerBoundSort[pos]=kd->lowerBound[src];
	}
}

__device__ __forceinline__
void countActive(cuStinger* custing,int32_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	if (kd->upperBound[src] > kd->lowerBound[kd->vertexArray[kd->K-1]]) {
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
		printf("\n@ %p %p %p %p %p %p %p %p @\n",kd->nPathsData,kd->nPaths, kd->nPathsPrev, kd->nPathsCurr, kd->KC,kd->lowerBound,kd->lowerBoundSort,kd->upperBound);
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
