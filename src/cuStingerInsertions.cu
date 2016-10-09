#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.hpp"


using namespace std;

// No duplicates allowed
__global__ void deviceUpdatesSweep1(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock){
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

		length_t upv = custing->dVD->getUsed()[src];		
		length_t epv = custing->dVD->getMax()[src];

		// if(src==140 && threadIdx.x==0)
		// 	printf("### %d %d %d \n",upv,epv,pos);

		// Checking to see if the edge already exists in the graph. 
		for (length_t e=threadIdx.x; e<srcInitSize && *found==0; e+=blockDim.x){
			if(d_adj[src]->dst[e]==dst){
				*found=1;
				break;
			}
		}
		__syncthreads();

		// If it does not exist, then it needs to be added.
		if(*found==0 && threadIdx.x==0){
			// IMPORTANT NOTE 
			// The following search for duplicates is rather safe and  should work most of the time.
			// There is an extreme and very unlikely case where this might fail.
			// This scenario can occur when the second edge block looks for duplicates however (**) has
			// not been completed/committed. Thus, it will not find a duplicate because d_adj[src]->dst[ret] is not set.
			// Also note, that this exact change is repeated in the second sweep.

			// Requesting a spot for insertion.
			length_t ret =  atomicAdd(d_utilized+src, 1);
			// Checking that there is enough space to insert this edge
			if(ret<d_max[src]){
				d_adj[src]->dst[ret] = dst;							// (**)
				length_t dupInBatch=0;

				// There might be an identical edge in the batch and we want to avoid adding it twice.
				// We are checking if possibly a different thread might have added it.
				for(length_t k=srcInitSize; k<ret; k++){
					if (d_adj[src]->dst[k]==dst)
						dupInBatch=1;
				}
				if(dupInBatch){
					// Duplicate edge in the batch. A redundant edge has been allocated in the used array for this edge.
					length_t duplicateID =  atomicAdd(d_dupCount, 1);
					d_indDuplicate[duplicateID] = pos;
					d_dupRelPos[duplicateID] = ret;
				}
			}
			else{
				// Out of space for this edge.				
				atomicSub(d_utilized+src,1);
				length_t inCompleteEdgeID =  atomicAdd(d_incCount, 1);
				d_indIncomplete[inCompleteEdgeID] = pos;
			}
		}
	}
}

__global__ void deviceUpdatesSweep2(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock){
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

	// All remaining updates will be processed.
	int32_t init_pos = blockIdx.x * updatesPerBlock;
	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=d_incCount[0])
			break;
		length_t indInc = d_indIncomplete[pos];
		vertexId_t src = d_updatesSrc[indInc],dst = d_updatesDst[indInc];
		length_t srcInitSize = d_utilized[src];

		if(threadIdx.x==0)
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
		// If it does not exist, then it needs to be added.
		if(*found==0 && threadIdx.x==0){
			// Requesting a spot for insertion.			
			length_t ret =  atomicAdd(d_utilized+src, 1);
			d_adj[src]->dst[ret] = dst;
			if(ret<d_max[src]){
				length_t dupInBatch=0;
				// There might be an identical edge in the batch and we want to avoid adding it twice.
				// We are checking if possibly a different thread might have added it.
				for(length_t k=srcInitSize; k<ret; k++){
					if (d_adj[src]->dst[k]==dst)
						dupInBatch=1;
				}
				if(dupInBatch){
					// Duplicate edge in the batch. A redundant edge has been allocated in the used array for this edge.					
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

	length_t* d_utilized      = custing->dVD->getUsed();
	// vertexId_t** d_adj          = custing->dVD->getAdj();	
	cuStinger::cusEdgeData** d_adj = custing->dVD->getAdj();	

	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();
	length_t* d_indDuplicate    = bud->getIndDuplicate();
	length_t* d_dupCount        = bud->getDuplicateCount();
	length_t* d_dupRelPos       = bud->getDupPosBatch();

	int32_t init_pos = blockIdx.x * dupsPerBlock;

	// Removing all duplicate edges that were found in the previous sweeps.
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
				d_adj[src]->dst[relPos] = d_adj[src]->dst[ret-1];
			}
		}
	}
}


void cuStinger::edgeInsertions(BatchUpdate &bu,length_t& requireAllocation){
	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t updatesPerBlock,dupsPerBlock;
	length_t updateSize,dupInBatch;

	requireAllocation=0;
	updateSize = *(bu.getHostBUD()->getBatchSize());
	numBlocks.x = ceil((float)updateSize/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	updatesPerBlock = ceil(float(updateSize)/float(numBlocks.x));

	deviceUpdatesSweep1<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
	checkLastCudaError("Error in the first update sweep");

	bu.getHostBUD()->copyDeviceToHostDupCount(*bu.getDeviceBUD());

	dupInBatch = *(bu.getHostBUD()->getDuplicateCount());
	// cout << "The number of duplicates in the batch is : " << dupInBatch << endl;
	if(dupInBatch>0){
		numBlocks.x = ceil((float)dupInBatch/(float)threads);
		if (numBlocks.x>1000){
			numBlocks.x=1000;
		}	
		dupsPerBlock = ceil(float(dupInBatch)/float(numBlocks.x));
		deviceRemoveInsertedDuplicates<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),dupsPerBlock);
		checkLastCudaError("Error in the first duplication sweep");
	}

	bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	reAllocateMemoryAfterSweep1(bu,requireAllocation);

	//--------
	// Sweep 2
	//--------

	updateSize = *(bu.getHostBUD()->getIncCount());
	bu.getDeviceBUD()->resetDuplicateCount();

	// if(false)
	if(updateSize>0){
		numBlocks.x = ceil((float)updateSize/(float)threads);
		if (numBlocks.x>16000){
			numBlocks.x=16000;
		}	
		updatesPerBlock = ceil(float(updateSize)/float(numBlocks.x));

		deviceUpdatesSweep2<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
		checkLastCudaError("Error in the second update sweep");
	
		bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
		dupInBatch = *(bu.getHostBUD()->getDuplicateCount());

		// cout << "The number of duplicates in the batch is : " << dupInBatch << endl;

		if(dupInBatch>0){
			numBlocks.x = ceil((float)dupInBatch/(float)threads);
			if (numBlocks.x>1000){
				numBlocks.x=1000;
			}	
			dupsPerBlock = ceil(float(dupInBatch)/float(numBlocks.x));
			deviceRemoveInsertedDuplicates<<<numBlocks,threadsPerBlock>>>(this->devicePtr(),
										 bu.getDeviceBUD()->devicePtr()	,dupsPerBlock);
			checkLastCudaError("Error in the second duplication sweep");
		}
	}

	bu.getHostBUD()->resetIncCount();
	bu.getDeviceBUD()->resetIncCount();
	bu.getHostBUD()->resetDuplicateCount();
	bu.getDeviceBUD()->resetDuplicateCount();
}


__global__ void deviceVerifyInsertions(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock, length_t* updateCounter){
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


bool cuStinger::verifyEdgeInsertions(BatchUpdate &bu)
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

	length_t* devCounter = (length_t*)allocDeviceArray(numBlocks.x,sizeof(length_t));

	deviceVerifyInsertions<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock,devCounter);

	length_t verified = cuStinger::sumDeviceArray(devCounter, numBlocks.x);

	freeDeviceArray(devCounter);

	if (verified==updateSize){
		return true;
	}
	else
		return false;

}

/*__global__ void deviceUpdatesSpace(cuStinger* custing, BatchUpdateData* bud){
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

	int32_t init_pos = blockIdx.x * updatesPerBlock;

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

		length_t upv = custing->dVD->getUsed()[src];		
		length_t epv = custing->dVD->getMax()[src];

		// if(src==140 && threadIdx.x==0)
		// 	printf("### %d %d %d \n",upv,epv,pos);

		// Checking to see if the edge already exists in the graph. 
		for (length_t e=threadIdx.x; e<srcInitSize && *found==0; e+=blockDim.x){
			if(d_adj[src]->dst[e]==dst){
				*found=1;
				break;
			}
		}
		__syncthreads();

		// If it does not exist, then it needs to be added.
		if(*found==0 && threadIdx.x==0){
			// IMPORTANT NOTE 
			// The following search for duplicates is rather safe and  should work most of the time.
			// There is an extreme and very unlikely case where this might fail.
			// This scenario can occur when the second edge block looks for duplicates however (**) has
			// not been completed/committed. Thus, it will not find a duplicate because d_adj[src]->dst[ret] is not set.
			// Also note, that this exact change is repeated in the second sweep.

			// Requesting a spot for insertion.
			length_t ret =  atomicAdd(d_utilized+src, 1);
			// Checking that there is enough space to insert this edge
			if(ret<d_max[src]){
				d_adj[src]->dst[ret] = dst;							// (**)
				length_t dupInBatch=0;

				// There might be an identical edge in the batch and we want to avoid adding it twice.
				// We are checking if possibly a different thread might have added it.
				for(length_t k=srcInitSize; k<ret; k++){
					if (d_adj[src]->dst[k]==dst)
						dupInBatch=1;
				}
				if(dupInBatch){
					// Duplicate edge in the batch. A redundant edge has been allocated in the used array for this edge.
					length_t duplicateID =  atomicAdd(d_dupCount, 1);
					d_indDuplicate[duplicateID] = pos;
					d_dupRelPos[duplicateID] = ret;
				}
			}
			else{
				// Out of space for this edge.				
				atomicSub(d_utilized+src,1);
				length_t inCompleteEdgeID =  atomicAdd(d_incCount, 1);
				d_indIncomplete[inCompleteEdgeID] = pos;
			}
		}
	}
}*/

__device__ void mergeYintoX(vertexId_t* X, vertexId_t const * Y, length_t lx, length_t ly)
{
	vertexId_t xcur = lx-1, ycur = ly-1, comp;
	while (xcur >= 0 && ycur >= 0){
		comp = X[xcur] - Y[ycur];
		X[xcur + ycur + 1] = (comp > 0)? X[xcur]:Y[ycur];
		xcur -= (comp > 0);
		ycur -= !(comp > 0);
	}
	for (int i = ycur; i >= 0; --i) {
        X[i] = Y[i];
    }
}

__global__ void deviceUpdatesMerge(cuStinger* custing, length_t const * const __restrict__ d_off,
	vertexId_t const * const __restrict__ d_ind, length_t const nv)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// if (tid == 1)
	// {
	// 	for (int i = 0; i < custing->dVD->getUsed()[tid]; ++i)
	// 	{
	// 		printf("%d\n", custing->dVD->getAdj()[tid]->dst[i]);
	// 	}
	// 	printf("============%d\n", d_off[tid+1] - d_off[tid]);
	// }
	if (tid < nv) {
		cuStinger::cusEdgeData** d_adj = custing->dVD->getAdj();
		mergeYintoX(d_adj[tid]->dst, &d_ind[d_off[tid]], custing->dVD->getUsed()[tid], d_off[tid+1] - d_off[tid]);
		custing->dVD->getUsed()[tid] += d_off[tid+1] - d_off[tid];
	}
	// if (tid == 1)
	// {
	// 	for (int i = 0; i < custing->dVD->getUsed()[tid]; ++i)
	// 	{
	// 		printf("%d\n", custing->dVD->getAdj()[tid]->dst[i]);
	// 	}
	// 	printf("============%d\n", d_off[tid+1] - d_off[tid]);
	// 	for (int i = d_off[tid]; i < d_off[tid+1] ; ++i)
	// 	{
	// 		printf("%d\n", d_ind[i]);
	// 	}
	// }
}

void cuStinger::edgeInsertionsSorted(length_t const * const __restrict__ d_off,
	vertexId_t const * const __restrict__ d_ind, length_t const nv, length_t& requireAllocation)
{
	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);

	// requireAllocation=0;

	// length_t *updateInc = (length_t*)allocDeviceArray(nv,sizeof(length_t));
	// numBlocks.x = ceil((float)nv/(float)threads);
	// deviceUpdatesSpace<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr());
	// checkLastCudaError("Error in the first update sweep");

	// bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	// reAllocateMemory(bu,requireAllocation);

	//--------
	// Sweep 2
	//--------

	numBlocks.x = ceil((float)nv/(float)threads);

	deviceUpdatesMerge<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), d_off, d_ind, nv);
	checkLastCudaError("Error in the second update sweep");
}