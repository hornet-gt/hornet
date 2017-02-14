#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils.hpp"
#include "update.hpp"
#include "memoryManager.hpp"
#include "cuStinger.hpp"

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

	/// Adding edges into the batch.
	deviceUpdatesSweep1<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
	checkLastCudaError("Error in the first update sweep");

	bu.getHostBUD()->copyDeviceToHostDupCount(*bu.getDeviceBUD());

	// Getting the number of edges in the batch that are duplicates and that numerous copies of that duplicate
	// edge where inserted into the graph,
	dupInBatch = *(bu.getHostBUD()->getDuplicateCount());

	// Removing any duplicate edges from the batch that were inserted multiple times.
	if(dupInBatch>0){
		numBlocks.x = ceil((float)dupInBatch/(float)threads);
		if (numBlocks.x>1000){
			numBlocks.x=1000;
		}	
		dupsPerBlock = ceil(float(dupInBatch)/float(numBlocks.x));
		deviceRemoveInsertedDuplicates<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),dupsPerBlock);
		checkLastCudaError("Error in the first duplication sweep");
	}

	// Some vertices may require additional space to store edges in their adjacency list.
	// Thus, we re-allocate memory and copy the elements.
	bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	reAllocateMemoryAfterSweep1(bu,requireAllocation);

	//--------
	// Sweep 2
	//--------

	updateSize = *(bu.getHostBUD()->getIncCount());
	bu.getDeviceBUD()->resetDuplicateCount();

	// if(false)
	// Running the 2nd sweep of the algorithm which takes care of the batch update edges that did not have enough memory. 
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

		// Getting the number of edges in the batch that are duplicates and that numerous copies of that duplicate
		// edge where inserted into the graph,
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


/// Checks that correctness of the insertion processs.
/// Goes through all the edges in the batch update and confirms that they are in graph.
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


/// Checks that correctness of the insertion processs and that all edges in the batch update appear in the graph.
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

__global__ void deviceUpdatesSpace(cuStinger* custing, BatchUpdateData* bud, int32_t updatesPerBlock, length_t* updateInc){
	length_t* d_utilized      = custing->dVD->getUsed();
	length_t* d_max           = custing->dVD->getMax();
	cuStinger::cusEdgeData** d_adj = custing->dVD->getAdj();	
	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();
	length_t batchSize          = *(bud->getBatchSize());
	length_t* d_incCount        = bud->getIncCount();
	vertexId_t* d_indIncomplete = bud->getIndIncomplete();
	length_t* d_indDuplicate    = bud->getIndDuplicate();

	__shared__ int32_t found[1];

	int32_t init_pos = blockIdx.x * updatesPerBlock;

	// Updates are processed one at a time	
	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=batchSize)
			break;

		if (d_indDuplicate[pos]==1) // this means it's a duplicate edge
			continue;
		// And just like that, we don't need to keep checking if there
		// is a duplicate in the batch. We're gonna skip right over it.

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

		// If it does not exist, then it needs to be added. (Not right now tho)
		// Ask for space to put it in. If not available, we'll allocate it later.
		if(*found==0 && threadIdx.x==0){

			// Requesting a spot for insertion.
			length_t ret =  atomicAdd(updateInc+src, 1);
			// Checking that there is enough space to insert this edge
			if((ret+d_utilized[src]+1)>d_max[src]){
				// Out of space for this edge.
				length_t inCompleteEdgeID =  atomicAdd(d_incCount, 1);
				d_indIncomplete[inCompleteEdgeID] = pos;
			}
		}
	}
}

__device__ void mergeYintoX(vertexId_t* X, vertexId_t const * Y, length_t lx, length_t ly, vertexId_t *Y_, length_t numDups)
{
	length_t xcur = lx-1, ycur = ly-1;
	vertexId_t comp;
	int ymask;
	while (xcur >= 0 && ycur >= 0){
		ymask = Y_[ycur];
		comp = X[xcur] - Y[ycur];
		if (!ymask) X[xcur + ycur + 1 - numDups] = (comp > 0)? X[xcur]:Y[ycur];
		xcur -= (comp >= 0 && !ymask);
		ycur -= (comp <= 0 || ymask);
		numDups -= ymask;
	}
	while (ycur >= 0) {
        if (!Y_[ycur]) X[xcur + ycur + 1 - numDups] = Y[ycur];
        numDups -= Y_[ycur];
        ycur--;
    }
}

__device__ void intersectCount(vertexId_t const*const uNodes, vertexId_t const*const vNodes,
	length_t uLength, length_t vLength, vertexId_t *vMask, length_t *numDups)
{
    int comp;
    int uCurr = 0;
    int vCurr = 0;
    int mask;
    while((vCurr < vLength) && (uCurr < uLength)){
    	mask = vMask[vCurr];
        comp = uNodes[uCurr] - vNodes[vCurr];
        *numDups += (comp == 0 && !mask);
        vMask[vCurr] |= (comp == 0);
        uCurr += (comp <= 0 && !mask);
        vCurr += (comp >= 0 || mask);
    }
}

/// Marking duplicate edges within the batch.
/// Assumes that edges are non-weighted and should not be counted multiple times.
__global__ void markDuplicates(BatchUpdateData* bud)
{
	length_t batchsize = *(bud->getBatchSize());
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < batchsize-1) { 
		vertexId_t *Y = bud->getDst();
		vertexId_t *X = bud->getSrc();
		vertexId_t *Y_ = bud->getIndDuplicate();
	    Y_[tid] = (Y[tid] == Y[tid+1] && X[tid] == X[tid+1]);
	    // TODO: try using an if condition to reduce number of writes
	    atomicAdd( bud->getvNumDuplicates() + X[tid], Y_[tid] );
	}
}

// TODO: argument list reverse (cuStinger* custing, BatchUpdateData* bud)
__global__ void deviceUpdatesMerge(BatchUpdateData* bud, cuStinger* custing)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	length_t nv = *(bud->getNumVertices());
	if (tid < nv) {
		cuStinger::cusEdgeData** d_adj = custing->dVD->getAdj();
		length_t *numDups = bud->getvNumDuplicates() + tid;
		length_t *d_off = bud->getOffsets();
		vertexId_t *d_ind = bud->getDst();

		intersectCount(d_adj[tid]->dst, &d_ind[d_off[tid]], custing->dVD->getUsed()[tid], d_off[tid+1] - d_off[tid], &(bud->getIndDuplicate()[d_off[tid]]), numDups);
		mergeYintoX(d_adj[tid]->dst, &d_ind[d_off[tid]], custing->dVD->getUsed()[tid], d_off[tid+1] - d_off[tid], &(bud->getIndDuplicate()[d_off[tid]]), *numDups);
		int increment = (d_off[tid+1] - d_off[tid] - *numDups);
		if (increment + custing->dVD->getUsed()[tid] > custing->dVD->getMax()[tid])
			printf("This shouldn't have happened. We allocated enough space %d\n", tid);
		custing->dVD->getUsed()[tid] += increment;
	}
}

__global__ void testDups(BatchUpdateData* bud)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	length_t batchsize = *(bud->getBatchSize());
	if (tid < batchsize)
	{
		if (bud->getIndDuplicate()[tid])
		{
			printf("%d\n", bud->getSrc()[tid]);
		}
	}
}

void cuStinger::edgeInsertionsSorted(BatchUpdate &bu, length_t& requireAllocation)
{
	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);

	length_t batchsize = *(bu.getHostBUD()->getBatchSize());
	numBlocks.x = ceil((float)batchsize/(float)threads);
	markDuplicates<<<numBlocks,threadsPerBlock>>>(bu.getDeviceBUD()->devicePtr());

	// TODO: Reallocation
	requireAllocation=0;

	length_t *updateInc = (length_t*)allocDeviceArray(nv,sizeof(length_t));
	cudaMemset(updateInc, 0, nv*sizeof(length_t));
	numBlocks.x = ceil((float)batchsize/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	int32_t updatesPerBlock = ceil(float(batchsize)/float(numBlocks.x));
	deviceUpdatesSpace<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(), updatesPerBlock, updateInc);
	checkLastCudaError("Error in the first update sweep");

	// Test
	// if the number of updates is same
	// length_t *updateIncH = (length_t*)allocHostArray(nv,sizeof(length_t));	
	// copyArrayDeviceToHost(updateInc, updateIncH, nv, sizeof(length_t));
	// int sum = 0;
	// for (int i = 0; i < nv; ++i)
	// {
	// 	sum+=updateIncH[i];
	// }
	// printf("sum %d\n", sum);

	bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	// printf("incCount = %d\n", *(bu.getHostBUD()->getIncCount()));
	reAllocateMemoryAfterSweep1(bu,requireAllocation);

	//--------
	// Sweep 2
	//--------

	numBlocks.x = (int)ceil((float)nv/(float)threads);
	deviceUpdatesMerge<<<numBlocks,threadsPerBlock>>>(bu.getDeviceBUD()->devicePtr(), this->devicePtr());
	cudaDeviceSynchronize();
	// checkLastCudaError("Error in the second update sweep");
	
	// // Test used number
	// length_t* h_dupcount = (length_t*) allocHostArray(batchsize, sizeof(length_t));
	// bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());

	// copyArrayHostToHost(bu.getHostBUD()->getIndDuplicate(), h_dupcount, batchsize, sizeof(length_t));
	// // printf("%p\n", bu.getDeviceBUD()->getvNumDuplicates());
	// // // cudaMemcpy(h_dupcount, bu.getDeviceBUD()->getvNumDuplicates(), sizeof(length_t)*nv, cudaMemcpyDeviceToHost);
	// printf("===\n");
	// for (int i = 0; i < batchsize; ++i)
	// {
	// 	count += h_dupcount[i];
	// 	if (h_dupcount[i] == 1)
	// 	{
	// 		printf("%d\n", bu.getHostBUD()->getSrc()[i]);
	// 	}
	// }
	// printf("%d\n", count);

	// numBlocks.x = ceil((float)batchsize/(float)threads);
	// testDups<<<numBlocks,threadsPerBlock>>>(bu.getDeviceBUD()->devicePtr());


	// freeHostArray(h_dupcount);	
	// numBlocks.x = ceil((float)batchsize/(float)threads);
	// markDuplicatesForward<<<numBlocks,threadsPerBlock>>>(bu.getDeviceBUD()->devicePtr(),testcount);
}