#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include "cct.hpp"

/*
	##     ##    ###     ######  ########   #######   ######
	###   ###   ## ##   ##    ## ##     ## ##     ## ##    ##
	#### ####  ##   ##  ##       ##     ## ##     ## ##
	## ### ## ##     ## ##       ########  ##     ##  ######
	##     ## ######### ##       ##   ##   ##     ##       ##
	##     ## ##     ## ##    ## ##    ##  ##     ## ##    ##
	##     ## ##     ##  ######  ##     ##  #######   ######
*/

// Typedef to vertex frontier expansion
typedef void (*cusSubKernel)(cuStinger* custing,vertexId_t src,void* metadata);

// High level macro definiton. To be used for stating the level of parallelism.
typedef __global__ void (*cusKernel)(cuStinger* custing,void* metadata, cusSubKernel* cusSK,int32_t verticesPerThreadBlock);

__global__ void allVerticesInGraph(cuStinger* custing,void* metadata, cusSubKernel* cusSK, int32_t verticesPerThreadBlock){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock+threadIdx.x;
	length_t nv = custing->getMaxNV();

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat+=blockDim.x){
		vertexId_t v=v_init+v_hat;
		if(v>=nv){
			break;
		}
		(*cusSK)(custing,v,metadata);
	}
}
__global__ void allVerticesInArray(vertexId_t* verArray, length_t len,cuStinger* custing,void* metadata, cusSubKernel* cusSK, int32_t verticesPerThreadBlock)
{
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		vertexId_t vpos=v_init+v_hat;
		if(vpos>=len){
			break;
		}
		vertexId_t v=verArray[vpos];
		(*cusSK)(custing,v,metadata);

	}
}




/*
	########  ########  ######
	##     ## ##       ##    ##
	##     ## ##       ##
	########  ######    ######
	##     ## ##             ##
	##     ## ##       ##    ##
	########  ##        ######
*/


typedef struct {
	vertexId_t* queue;
	length_t queueCurr;
	length_t queueEnd;
	vertexId_t* level;
	vertexId_t currLevel;
}bfsData;



__device__ void bfsExpandFrontier(cuStinger* custing,vertexId_t src, void* metadata){
	bfsData* bd = (bfsData*)metadata;

	length_t srcLen=custing->dVD->used[src];
	vertexId_t* adj_src=custing->dVD->adj[src]->dst;
	// length_t srcLen=custing->dVD->getUsed()[src];
	// vertexId_t* adj_src=custing->dVD->getAdj()[src]->dst;

	vertexId_t nextLevel=bd->currLevel+1;
	for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
		vertexId_t dest = adj_src[adj];
		vertexId_t prev = atomicCAS(bd->level+dest,INT32_MAX,nextLevel);
		if(prev==INT32_MAX){
			length_t prevPos = atomicAdd(&(bd->queueEnd),1);
			bd->queue[prevPos] = dest;
		}
	}
}
__device__ cusSubKernel ptrBFSExpandFrontier = bfsExpandFrontier;

__device__ void setLevelInfinity(cuStinger* custing,vertexId_t src, void* metadata){
	bfsData* bd = (bfsData*)metadata;
	bd->level[src]=INT32_MAX;
}
__device__ cusSubKernel ptrSetLevelInfinity = setLevelInfinity;



void callkernel(cuStinger& custing)
{
	cudaEvent_t ce_start,ce_stop;	

	start_clock(ce_start, ce_stop);


	bfsData hostBfsData;
	hostBfsData.queue = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostBfsData.level = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostBfsData.queueCurr=0;
    hostBfsData.queueEnd=1;
	hostBfsData.currLevel=0;

	bfsData* deviceBfsData = (bfsData*)allocDeviceArray(1, sizeof(bfsData));
	copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));

	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock;

	numBlocks.x = ceil((float)custing.nv/(float)threads);
	if (numBlocks.x>64000){
		numBlocks.x=64000;
	}	
	verticesPerThreadBlock = ceil(float(custing.nv)/float(numBlocks.x-1));



	cusSubKernel* dSetInfinity2 = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dSetInfinity2, ptrSetLevelInfinity, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	allVerticesInGraph<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceBfsData,dSetInfinity2,verticesPerThreadBlock);
	freeDeviceArray(dSetInfinity2);

	vertexId_t root=2; length_t level=0;
	copyArrayHostToDevice(&root,hostBfsData.queue,1,sizeof(vertexId_t));
	copyArrayHostToDevice(&level,hostBfsData.level+root,1,sizeof(length_t));

	cusSubKernel* dTraverseEdges = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dTraverseEdges, ptrBFSExpandFrontier, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	// cudaMemcpyFromSymbol( dTraverseEdges, ptrBFSExpandFrontier, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);

	length_t prevEnd=1;
	while(hostBfsData.queueEnd-hostBfsData.queueCurr>0){
		allVerticesInArray<<<numBlocks, threadsPerBlock>>>(hostBfsData.queue+hostBfsData.queueCurr,
										hostBfsData.queueEnd-hostBfsData.queueCurr,custing.devicePtr(),
										deviceBfsData,dTraverseEdges,verticesPerThreadBlock);
		copyArrayDeviceToHost(deviceBfsData,&hostBfsData,1, sizeof(bfsData));

		hostBfsData.queueCurr=prevEnd;
		prevEnd = hostBfsData.queueEnd;
		hostBfsData.currLevel++;
		copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));

	}
	cout << "The queue end  :" << hostBfsData.queueEnd << endl;

		freeDeviceArray(dTraverseEdges);

	float totalBFSTime = end_clock(ce_start, ce_stop);
	cout << "Total time for the BFS : " << totalBFSTime << endl; 


	freeDeviceArray(deviceBfsData);
	freeDeviceArray(hostBfsData.queue);
	freeDeviceArray(hostBfsData.level);
}





