#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.hpp"


using namespace std;

// No duplicates allowed
__global__ void deviceUpdatesSweep1(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock){
	length_t* d_utilized      = custing->getDeviceUtilized();
	length_t* d_max           = custing->getDeviceMax();
	vertexId_t** d_adj          = custing->getDeviceAdj();	
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

	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=batchSize)
			break;
		vertexId_t src = d_updatesSrc[pos];
		vertexId_t dst = d_updatesDst[pos];

		length_t srcInitSize = d_utilized[src];
		if(threadIdx.x ==0)
			*found=0;
		__syncthreads();

		for (length_t e=threadIdx.x; e<srcInitSize; e+=blockDim.x){
			if(d_adj[src][e]==dst){
				*found=1;
			}
		}
		__syncthreads();
		if(!(*found) && threadIdx.x==0){
			length_t ret =  atomicAdd(d_utilized+src, 1);
			if(ret<d_max[src]){
				length_t dupInBatch=0;
				for(length_t k=srcInitSize; k<ret; k++){
					if (d_adj[src][k]==dst)
						dupInBatch=1;
				}
				if(!dupInBatch){
					d_adj[src][ret] = dst;
				}
				else{
					length_t duplicateID =  atomicAdd(d_dupCount, 1);
					d_indDuplicate[duplicateID] = pos;
					d_dupRelPos[duplicateID] = ret;

				}
			}
			else{
				atomicSub(d_utilized+src,1);
				// Out of space for this adjacency.
				length_t inCompleteEdgeID =  atomicAdd(d_incCount, 1);
				d_indIncomplete[inCompleteEdgeID] = pos;
			}
		}
	}
}

__global__ void deviceUpdatesSweep2(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock){
	length_t* d_utilized      = custing->getDeviceUtilized();
	length_t* d_max           = custing->getDeviceMax();
	vertexId_t** d_adj          = custing->getDeviceAdj();	
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

	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=d_incCount[0])
			break;
		length_t indInc = d_indIncomplete[pos];
		vertexId_t src = d_updatesSrc[indInc];
		vertexId_t dst = d_updatesDst[indInc];

		length_t srcInitSize = d_utilized[src];

		if(threadIdx.x==0)
			*found=0;
		__syncthreads();

		for (length_t e=threadIdx.x; e<srcInitSize; e+=blockDim.x){
			if(d_adj[src][e]==dst){
				*found=1;
			}
		}
		__syncthreads();

		if(!(*found) && threadIdx.x==0){
			length_t ret =  atomicAdd(d_utilized+src, 1);
			if(ret<d_max[src]){
				length_t dupInBatch=0;
				for(length_t k=srcInitSize; k<ret; k++){
					if (d_adj[src][k]==dst)
						dupInBatch=1;
				}
				if(!dupInBatch){
					d_adj[src][ret] = dst;
				}
				else{
					length_t duplicateID =  atomicAdd(d_dupCount, 1);
					d_indDuplicate[duplicateID] = pos;
					d_dupRelPos[duplicateID] = ret;
				}
			}
			else{
				printf("This should never happen because of reallaction");
			}
		}
	}
}

// Currently using a single thread in the warp for duplicate edge removal
__global__ void deviceRemoveInsertedDuplicates(cuStinger* custing, BatchUpdateData* bud,int32_t dupsPerBlock){

	length_t* d_utilized      = custing->getDeviceUtilized();
	vertexId_t** d_adj          = custing->getDeviceAdj();	

	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();
	length_t* d_indDuplicate    = bud->getIndDuplicate();
	length_t* d_dupCount        = bud->getDuplicateCount();
	length_t* d_dupRelPos       = bud->getDupPosBatch();


	int32_t init_pos = blockIdx.x * dupsPerBlock;

	for (int32_t i=0; i<dupsPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=d_dupCount[0])	
			break;
		if (threadIdx.x==0){
			length_t indDup = d_indDuplicate[pos];
			vertexId_t src    = d_updatesSrc[indDup];
			length_t relPos = d_dupRelPos[indDup];

			length_t ret =  atomicSub(d_utilized+src, 1);
			if(ret>0){
				d_adj[src][relPos] = d_adj[src][ret-1];
			}
		}
	}
}


void update(cuStinger &custing, BatchUpdate &bu)
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
	updatesPerBlock = ceil(float(updateSize)/float(numBlocks.x-1));

	// deviceUpdatesSweep1<<<numBlocks,threadsPerBlock>>>(custing.devicePtr(), bu.devicePtr(),updatesPerBlock);
	deviceUpdatesSweep1<<<numBlocks,threadsPerBlock>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
	checkLastCudaError("Error in the first update sweep");

	bu.getHostBUD()->copyDeviceToHostDupCount(*bu.getDeviceBUD());
	// bu.copyDeviceToHostDupCount();
//	dupInBatch = bu.getHostDuplicateCount();
	dupInBatch = *(bu.getHostBUD()->getDuplicateCount());
	cout << "The number of duplicates in the batch is : " << dupInBatch << endl;
	if(dupInBatch>0){
		numBlocks.x = ceil((float)dupInBatch/(float)threads);
		if (numBlocks.x>1000){
			numBlocks.x=1000;
		}	
		dupsPerBlock = ceil(float(dupInBatch)/float(numBlocks.x-1));
		deviceRemoveInsertedDuplicates<<<numBlocks,threadsPerBlock>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(),dupsPerBlock);
		checkLastCudaError("Error in the first duplication sweep");
	}

	bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	bu.reAllocateMemoryAfterSweep1(custing);
	
	// return;
	//--------
	// Sweep 2
	//--------
	// cout << "ODED YOU STILL NEED to add back some additional functionality below into BU" << endl;	
	// bu.copyDeviceToHostIncCount();
	// updateSize = bu.getHostIncCount();
	// bu.resetDeviceDuplicateCount();

	updateSize = *(bu.getHostBUD()->getIncCount());
	bu.getDeviceBUD()->resetDuplicateCount();

	if(updateSize>0){
		numBlocks.x = ceil((float)updateSize/(float)threads);
		if (numBlocks.x>16000){
			numBlocks.x=16000;
		}	
		updatesPerBlock = ceil(float(updateSize)/float(numBlocks.x-1));

		deviceUpdatesSweep2<<<numBlocks,threadsPerBlock>>>(custing.devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
		checkLastCudaError("Error in the second update sweep");
	
		bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
		dupInBatch = *(bu.getHostBUD()->getDuplicateCount());

		cout << "The number of duplicates in the batch is : " << dupInBatch << endl;

		if(dupInBatch>0){
			numBlocks.x = ceil((float)dupInBatch/(float)threads);
			if (numBlocks.x>1000){
				numBlocks.x=1000;
			}	
			dupsPerBlock = ceil(float(dupInBatch)/float(numBlocks.x-1));
			deviceRemoveInsertedDuplicates<<<numBlocks,threadsPerBlock>>>(custing.devicePtr(),
										 bu.getDeviceBUD()->devicePtr()	,dupsPerBlock);
			checkLastCudaError("Error in the second duplication sweep");
		}
	}

	// cout << "The number of duplicates in the second sweep : " << bu.getHostDuplicateCount() << endl;
	bu.getHostBUD()->resetIncCount();
	bu.getDeviceBUD()->resetIncCount();
	bu.getHostBUD()->resetDuplicateCount();
	bu.getDeviceBUD()->resetDuplicateCount();
}


