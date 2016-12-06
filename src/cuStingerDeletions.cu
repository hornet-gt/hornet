#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils.hpp"
#include "update.hpp"
#include "memoryManager.hpp"
#include "cuStinger.hpp"

using namespace std;

__global__ void deviceEdgeDeletesSweep1(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock){
	length_t* d_utilized      = custing->dVD->getUsed();
	length_t* d_max           = custing->dVD->getMax();
	cuStinger::cusEdgeData** d_adj = custing->dVD->getAdj();	
	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();
	length_t batchSize          = *(bud->getBatchSize());
	vertexId_t* d_indIncomplete = bud->getIndIncomplete();					

	__shared__ int64_t found[1], research[1];

	int32_t init_pos = blockIdx.x * updatesPerBlock;
	// Updates are processed one at a time	
	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=batchSize)
			break;
		__syncthreads();

		vertexId_t src = d_updatesSrc[pos],dst = d_updatesDst[pos];
		if(threadIdx.x ==0){
			*found=-1;
			d_indIncomplete[pos]=DELETION_MARKER;
		}
		__syncthreads();

		length_t srcInitSize = d_utilized[src];
		// Checking to see if the edge already exists in the graph. 
		for (length_t e=threadIdx.x; e<srcInitSize && *found==-1; e+=blockDim.x){
			if(d_adj[src]->dst[e]==dst){
				*found=e;
				break;
			}
		}
		__syncthreads();

		length_t last,dupLast;
		vertexId_t prevValCurr, prevValMove,lastVal;
		if(*found!=-1 && threadIdx.x==0){
			prevValCurr = atomicCAS(d_adj[src]->dst + *found,dst,DELETION_MARKER);
			if(prevValCurr!=DELETION_MARKER){
				d_indIncomplete[pos]=*found;
			}
		}
		__syncthreads();
	}
}

__global__ void deviceEdgeDeletesSweep2(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock){
	length_t* d_utilized      = custing->dVD->getUsed();
	length_t* d_max           = custing->dVD->getMax();
	cuStinger::cusEdgeData** d_adj = custing->dVD->getAdj();	
	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();
	length_t batchSize          = *(bud->getBatchSize());
	vertexId_t* d_indIncomplete = bud->getIndIncomplete();					

	__shared__ int32_t completed, needRepeat;


	int32_t init_pos = blockIdx.x * updatesPerBlock;
	// Updates are processed one at a time	
	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=batchSize)
			break;
		__syncthreads();

		if(d_indIncomplete[pos]==DELETION_MARKER){
			continue;
		}
		if(threadIdx.x==0)
			completed=-1;
		needRepeat=1;
		__syncthreads();

		vertexId_t src = d_updatesSrc[pos],dst = d_updatesDst[pos];		
		while(needRepeat==1){
			if(threadIdx.x==0)
				needRepeat=0;
			// Checking to see if the edge already exists in the graph. 
			for (length_t e=d_utilized[src]-threadIdx.x; e>d_indIncomplete[pos] && completed==-1; e-=blockDim.x){
				if(d_adj[src]->dst[e]!=DELETION_MARKER){
					atomicMax(&completed,e);
					break;
				}
				__syncthreads();
			}
			__syncthreads();

			if(completed!=-1 && threadIdx.x==0){
				length_t prevVal = atomicExch(d_adj[src]->dst + completed,DELETION_MARKER);
				if(prevVal!=DELETION_MARKER)
					d_adj[src]->dst[d_indIncomplete[pos]] = d_adj[src]->dst[completed];
				else{
					needRepeat = 1;
					completed = -1;
				}
			}
			__syncthreads();
		}
		if(threadIdx.x==0)
			atomicSub(d_utilized+src,1);
		__syncthreads();
	}
}


void cuStinger::edgeDeletions(BatchUpdate &bu)
{	
	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t updatesPerBlock,dupsPerBlock;
	length_t updateSize,dupInBatch;

	updateSize = *(bu.getHostBUD()->getBatchSize());
	numBlocks.x = ceil((float)updateSize/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	updatesPerBlock = ceil(float(updateSize)/float(numBlocks.x));

	deviceEdgeDeletesSweep1<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
	checkLastCudaError("Error in the first delete sweep");

	deviceEdgeDeletesSweep2<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
	checkLastCudaError("Error in the first delete sweep");

	// bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	// bu.getHostBUD()->resetIncCount();
	// bu.getDeviceBUD()->resetIncCount();
	// bu.getHostBUD()->resetDuplicateCount();
	// bu.getDeviceBUD()->resetDuplicateCount();
}


	
__global__ void deviceVerifyDeletions(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock, length_t* updateCounter){
	length_t* d_utilized      = custing->dVD->getUsed();
	length_t* d_max           = custing->dVD->getMax();
	cuStinger::cusEdgeData** d_adj = custing->dVD->getAdj();	
	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();
	length_t batchSize          = *(bud->getBatchSize());
	length_t* d_incCount        = bud->getIncCount();
	vertexId_t* d_indIncomplete = bud->getIndIncomplete();
	length_t* d_indDuplicate    = bud->getIndDuplicate();
	length_t* d_dupCount        = bud->getDuplicateCount();
	length_t* d_dupRelPos       = bud->getDupPosBatch();

	__shared__ int32_t found[1];

	int32_t init_pos = blockIdx.x * updatesPerBlock;

	if (threadIdx.x==0)
		updateCounter[blockIdx.x]=0;
	__syncthreads();

	// Updates are processed one at a time	
	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=batchSize)
			break;

		vertexId_t src = d_updatesSrc[pos],dst = d_updatesDst[pos];
		length_t srcInitSize = d_utilized[src];
		if(threadIdx.x ==0)
			*found=0;
		__syncthreads();

		// Checking to see if the edge already exists in the graph. 
		for (length_t e=threadIdx.x; e<srcInitSize && *found==0; e+=blockDim.x){
			if(d_adj[src]->dst[e]==dst){
				*found=1;
				break;
			}
		}
		__syncthreads();
	
		if (threadIdx.x==0)
			updateCounter[blockIdx.x]+=*found;
		__syncthreads();
	}
}

bool cuStinger::verifyEdgeDeletions(BatchUpdate &bu){
	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t updatesPerBlock,dupsPerBlock;
	length_t updateSize,dupInBatch;

	updateSize = *(bu.getHostBUD()->getBatchSize());
	numBlocks.x = ceil((float)updateSize/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	updatesPerBlock = ceil(float(updateSize)/float(numBlocks.x));

	length_t* devCounter = (length_t*)allocDeviceArray(numBlocks.x,sizeof(length_t));
	deviceVerifyDeletions<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock,devCounter);
	length_t verified = cuStinger::sumDeviceArray(devCounter, numBlocks.x);

	freeDeviceArray(devCounter);
	
	if (verified==0)
		return true;
	else
		return false;
}

