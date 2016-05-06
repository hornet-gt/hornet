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
			if(threadIdx.x==0)
				printf("*");
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

			if(completed!=-1 && threadIdx.x==0 && completed<d_indIncomplete[pos])			
				printf("*");
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
				// printf("^^^^ Found it : %d %d\n",src,dst);
				break;
			}
		}
		__syncthreads();
	
		if (threadIdx.x==0)
			updateCounter[blockIdx.x]+=*found;
		__syncthreads();
	}
}

void cuStinger::verifyEdgeDeletions(BatchUpdate &bu){
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
	cout << "Number of blocks in the deletion is : "  << numBlocks.x << endl;

	length_t* devCounter = (length_t*)allocDeviceArray(numBlocks.x,sizeof(length_t));
	deviceVerifyDeletions<<<numBlocks,threadsPerBlock>>>(this->devicePtr(), bu.getDeviceBUD()->devicePtr(),updatesPerBlock,devCounter);
	length_t verified = cuStinger::sumDeviceArray(devCounter, numBlocks.x);

	if (verified==0)
		cout << "All deletions are accounted for.             Not deleted : " << verified << endl;
	else
		cout << "Some of the deletions are NOT accounted for. Not deleted : " << verified << endl;

	freeDeviceArray(devCounter);
}





/*
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
		for(int iter=0; iter<10 && *found==-1; iter++){
			srcInitSize = max(srcInitSize,d_utilized[src]);

			// length_t srcInitSize = d_utilized[src];
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
				if(prevValCurr==DELETION_MARKER){
					d_indIncomplete[pos]=*found;
				}
			}

			// if(*found!=-1 && threadIdx.x==0){
			//  	if(d_utilized[src]!=srcInitSize){
			//  		*found=-1;
			//  	}
			//  	last =  atomicSub(d_utilized+src, 1)-1; // Recall that the utilized refers to the length, thus we need to subtract one to get the last element
			//  	if(*found<last){
			// 		prevValCurr = atomicCAS(d_adj[src]->dst + *found,dst,d_adj[src]->dst[last]);
			// 		if(prevValCurr!=dst){
			//  			*found=-1;
			//  			atomicAdd(d_utilized+src, 1);
			// 		}
			//  	}
			// 	else if(last==0){
			// 			dupLast =  atomicAdd(d_utilized+src, 1); // Recall that the utilized refers to the length, thus we need to subtract one to get the last element
			// 			*found=-1;
			// 			// printf("Possibly here\n");
			// 	}else if (last<0){ // Trying to delete when an edge doesn't exist.	
			// 			dupLast =  atomicAdd(d_utilized+src, 1); // Recall that the utilized refers to the length, thus we need to subtract one to get the last element
			// 			*found=-1;
			// 			// printf("Or Possibly here\n");
			// 	}

			// }


			// if(*found!=-1 && threadIdx.x==0){
			// 	last =  atomicSub(d_utilized+src, 1)-1; // Recall that the utilized refers to the length, thus we need to subtract one to get the last element
			// 	if(*found <last){
			// 		if(last>0){
			// 			// prevVal = atomicCAS(d_adj[src]->dst + *found,dst,lastVal);
			// 			prevValMove = atomicCAS(d_adj[src]->dst + last,d_adj[src]->dst[last],DELETION_MARKER);
			// 			if(prevValMove==DELETION_MARKER){// edge has already moved by another vertex
			// 				*found=-1;
			// 				atomicAdd(d_utilized+src, 1);
			// 				// iter--;
			// 			}
			// 			else{
			// 				prevValCurr = atomicCAS(d_adj[src]->dst + *found,dst,prevValMove);
			// 				if(prevValCurr==DELETION_MARKER){
			// 					*found=-1;
			// 					atomicCAS(d_adj[src]->dst + last,DELETION_MARKER,prevValMove);
			// 					atomicAdd(d_utilized+src, 1);
			// 				}
			// 				else if(prevValCurr!=dst){
			// 					*found=-1; 
			// 					atomicCAS(d_adj[src]->dst + last,DELETION_MARKER,prevValMove);
			// 					atomicAdd(d_utilized+src, 1);
			// 					// d_adj[src]->dst[dupLast] =lastVal;
			// 					//printf("For the love of me\n");
			// 				}
			// 				else{
			// 					atomicCAS(d_adj[src]->dst + last,DELETION_MARKER,dst);
			// 				}
			// 			}
			// 		}
			// 	}
			// 	else if(*found>=last){
			// 		prevValCurr = atomicCAS(d_adj[src]->dst + *found,dst,DELETION_MARKER);
			// 		if (prevValCurr!=dst)
			// 			printf("Ahh OHH\n");
			// 	}
			// 	else if(last==0){
			// 			dupLast =  atomicAdd(d_utilized+src, 1); // Recall that the utilized refers to the length, thus we need to subtract one to get the last element
			// 			*found=-1;
			// 			// printf("Possibly here\n");
			// 	}else if (last<0){ // Trying to delete when an edge doesn't exist.	
			// 			dupLast =  atomicAdd(d_utilized+src, 1); // Recall that the utilized refers to the length, thus we need to subtract one to get the last element
			// 			*found=-1;
			// 			// printf("Or Possibly here\n");
			// 	}
			// }
			__syncthreads();

#if 1
			// if(src ==26771 && dst ==30510 && threadIdx.x==0){
			if(iter ==9 && threadIdx.x==0){
				printf("\nI DID IT %d %d %d %d\n", src,dst, srcInitSize, d_utilized[src]);
				for(length_t e=0; e<srcInitSize; e++)
					printf("%d ,",d_adj[src]->dst[e]);
				printf("\n");
				for(length_t e=0; e<d_utilized[src]; e++)
					printf("%d ,",d_adj[src]->dst[e]);
				printf("\n");

				printf("\n");
			}
#endif

			__syncthreads();

		}

	}
}
*/