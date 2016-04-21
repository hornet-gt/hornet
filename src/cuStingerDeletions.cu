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
		length_t srcInitSize = d_utilized[src];
		if(threadIdx.x ==0){
			*found=-1;
			*research=1;
		}
		__syncthreads();

		int researchCounter=0;
		while(*research){
			*research=0;

			// Checking to see if the edge already exists in the graph. 
			for (length_t e=threadIdx.x; e<srcInitSize && *found==-1; e+=blockDim.x){
				if(d_adj[src]->dst[e]==dst){
					*found=e;
					break;
				}
			}

			__syncthreads();

			length_t ret,ret2;
			vertexId_t prevVal;

			if(*found==-1)
				printf("***");
			// If edge has been found, then it needs to be deleted.
			if(*found!=-1 && threadIdx.x==0){
				prevVal = atomicExch(d_adj[src]->dst + *found, DELETION_MARKER);
				// if (blockIdx.x==0)
				// 	printf("%d %d \n", prevVal,dst);
				if(prevVal==dst){
					// Requesting a spot for insertion.
					ret =  atomicSub(d_utilized+src, 1);

					if (ret>0 && *found > ret){
						ret2 = atomicAdd(d_utilized+src, 1);
						if(ret!=ret2){
							// d_adj[src]->dst[ret2] = d_adj[src]->dst[ret];
							// d_adj[src]->dst[ret] = DELETION_MARKER;
							printf("We gotta a problem\n");
						}
						*research=1;
					}
					else if(ret>0){
						d_adj[src]->dst[*found] = d_adj[src]->dst[ret];
						// d_adj[src]->dst[ret] = DELETION_MARKER;
					}
				}
				else{
					printf("#$");
				}

			}
			else{
				if(*found==-1 && threadIdx.x==0)
					printf("%ld", *found );

			}
			__syncthreads();

			researchCounter++;
			if(researchCounter>10){
				printf("Stuck in a deletion\n");
				break;
			}
		}

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
	cout << "ODED - the update size is :" << updateSize << endl;
	numBlocks.x = ceil((float)updateSize/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	updatesPerBlock = ceil(float(updateSize)/float(numBlocks.x-1));

	deviceEdgeDeletesSweep1<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock);
	checkLastCudaError("Error in the first delete sweep");

	bu.getHostBUD()->copyDeviceToHost(*bu.getDeviceBUD());
	reAllocateMemoryAfterSweep1(bu);


	bu.getHostBUD()->resetIncCount();
	bu.getDeviceBUD()->resetIncCount();
	bu.getHostBUD()->resetDuplicateCount();
	bu.getDeviceBUD()->resetDuplicateCount();
}


	
