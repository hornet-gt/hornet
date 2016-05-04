#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.hpp"

using namespace std;

__global__ void deviceEdgeDeletesSweep1(cuStinger* custing, BatchUpdateData* bud,int32_t updatesPerBlock){
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
		}

		__syncthreads();

		for(int iter=0; iter<10 && *found==-1; iter++){
			length_t srcInitSize = d_utilized[src];

			// Checking to see if the edge already exists in the graph. 
			for (length_t e=threadIdx.x; e<srcInitSize && *found==-1; e+=blockDim.x){
				if(d_adj[src]->dst[e]==dst){
					*found=e;
					break;
				}
			}
			__syncthreads();
	
			length_t ret;
			vertexId_t prevVal;

			if(*found!=-1 && threadIdx.x==0){
				prevVal = atomicCAS(d_adj[src]->dst + *found,dst,DELETION_MARKER);

				if(prevVal!=DELETION_MARKER){
					ret =  atomicSub(d_utilized+src, 1);

					if(ret>0){
						d_adj[src]->dst[*found]=d_adj[src]->dst[ret-1];
						d_adj[src]->dst[ret-1]=DELETION_MARKER;
						// if(d_adj[src]->dst[ret]==DELETION_MARKER){
						// 	printf("Dup in deletion\n");
						// }
						// if(src==9638)
						// 	printf("Here\n");
					}
					else{
						printf("Nothing left\n");
					}
				}
				else{
					printf("I beat to the deletion\n");
				}
			}
			// else if (*found==-1){
			// 	// Checking to see if the edge already exists in the graph. 
			// 	// for (length_t e=threadIdx.x; e<d_max[src]; e+=blockDim.x){
			// 	// 	if(d_adj[src]->dst[e]==dst){
			// 	// 		printf("AAGOOO\n");
			// 	// 		break;
			// 	// 	}
			// 	// }
			// }
			__syncthreads();
		}

	}
}


/*
			// If edge has been found, then it needs to be deleted.
			// if(*found!=-1 && threadIdx.x==0){
			// 	prevVal = atomicExch(d_adj[src]->dst + *found, DELETION_MARKER);
			// 	// if (blockIdx.x==0)
			// 	// 	printf("%d %d \n", prevVal,dst);
			// 	if(prevVal==dst){
			// 		// Requesting a spot for insertion.

			// 		if (ret>0 && *found > ret){
			// 			ret2 = atomicAdd(d_utilized+src, 1);
			// 			if(ret!=ret2){
			// 				// d_adj[src]->dst[ret2] = d_adj[src]->dst[ret];
			// 				// d_adj[src]->dst[ret] = DELETION_MARKER;
			// 				printf("We gotta a problem\n");
			// 			}
			// 			*research=1;
			// 		}
			// 		else if(ret>0){
			// 			d_adj[src]->dst[*found] = d_adj[src]->dst[ret];
			// 			// d_adj[src]->dst[ret] = DELETION_MARKER;
			// 		}
			// 	}
			// 	else{
			// 		printf("#$");
			// 	}

			// }
			// else{
			// 	if(*found==-1 && threadIdx.x==0)
			// 		printf("%ld", *found );

			// }
			// __syncthreads();

*/


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

	// cout << "Deletions : " << updatesPerBlock<< endl;
	// cout << "Deletions : " << numBlocks.x << endl;
	// cout << "Deletions : " << threadsPerBlock.x << endl;

	deviceEdgeDeletesSweep1<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
	checkLastCudaError("Error in the first delete sweep");

	bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	reAllocateMemoryAfterSweep1(bu);

	bu.getHostBUD()->resetIncCount();
	bu.getDeviceBUD()->resetIncCount();
	bu.getHostBUD()->resetDuplicateCount();
	bu.getDeviceBUD()->resetDuplicateCount();
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
			// if(d_adj[src]->dst[e]==DELETION_MARKER){
			// 	printf("DELETIONS are not smooth\n");
			// 	break;
			// }

		}
		__syncthreads();
	
		if (threadIdx.x==0)
			updateCounter[blockIdx.x]+=*found;
		__syncthreads();

	}
}


void cuStinger::verifyEdgeDeletions(BatchUpdate &bu)
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

	deviceVerifyDeletions<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock,devCounter);

	length_t verified = cuStinger::sumDeviceArray(devCounter, numBlocks.x);

	if (verified==0)
		cout << "All deletions are accounted for.             Not deleted : " << verified << endl;
	else
		cout << "Some of the deletions are NOT accounted for. Not deleted : " << verified << endl;

	freeDeviceArray(devCounter);
}



