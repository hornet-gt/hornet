#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>


// #include "moderngpu.cuh"
// #include "moderngpu/src/kernel_scan.hxx"
#include "kernel_scan.hxx"
#include "search.hxx"
#include "cta_search.hxx"

// ModernGPU
#include <kernel_sortedsearch.hxx>
#include <transform.hxx>
using namespace mgpu;

//CUB headers
#include <cub.cuh>
using namespace cub;


#include <block/block_radix_sort.cuh>



#include <iostream>

#include "cct.hpp"


typedef struct {
	vertexId_t* queue;
	length_t queueCurr;
	length_t queueEnd;
	vertexId_t* currState;
	vertexId_t* prevState;
	vertexId_t changeCurrIter;
	vertexId_t connectedComponentCount;
}connectedComponentsData;



/*
	##     ##    ###     ######  ########   #######   ######
	###   ###   ## ##   ##    ## ##     ## ##     ## ##    ##
	#### ####  ##   ##  ##       ##     ## ##     ## ##
	## ### ## ##     ## ##       ########  ##     ##  ######
	##     ## ######### ##       ##   ##   ##     ##       ##
	##     ## ##     ## ##    ## ##    ##  ##     ## ##    ##
	##     ## ##     ##  ######  ##     ##  #######   ######
*/

// #define CUSTINGER_EDGE_SOURCE      source__
// #define CUSTINGER_EDGE_TYPE        type__
#define CUSTINGER_EDGE_DEST        dest__
// #define CUSTINGER_EDGE_WEIGHT      weight__
// #define CUSTINGER_EDGE_TIME_FIRST  timeFirst__
// #define CUSTINGER_EDGE_TIME_RECENT timeRecent__


#define CUSTINGER_FOR_ALL_EDGES_OF_VERTEX_PAR_THREAD_BLOCK_BEGIN(cuSTINGER, v) \
	length_t srcLen=cuSTINGER->dVD->used[src]; \
	vertexId_t* adj_src=cuSTINGER->dVD->adj[src]->dst; \
	for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){\
		vertexId_t dest__ = adj_src[adj];

#define CUSTINGER_FOR_ALL_EDGES_OF_VERTEX_PAR_THREAD_BLOCK_END() \
	}



// Typedef to vertex frontier expansion
typedef void (*cusSubKernel)(cuStinger* custing,vertexId_t src,void* metadata);

// High level macro definiton. To be used for stating the level of parallelism.
typedef  void (*cusKernel)(cuStinger* custing,void* metadata, cusSubKernel* cusSK,int32_t verticesPerThreadBlock);

__global__ void allVerticesInGraphParallelVertexPerThreadBlock(cuStinger* custing,void* metadata, cusSubKernel* cusSK, int32_t verticesPerThreadBlock){
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

__global__ void allVerticesInGraphOneVertexPerThreadBlock(cuStinger* custing,void* metadata, cusSubKernel* cusSK, int32_t verticesPerThreadBlock){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock;
	length_t nv = custing->getMaxNV();

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		vertexId_t v=v_init+v_hat;
		if(v>=nv){
			break;
		}
		(*cusSK)(custing,v,metadata);
	}
}

__global__ void allVerticesInArrayParallelVertexPerTB(vertexId_t* verArray, length_t len,cuStinger* custing,void* metadata, cusSubKernel* cusSK, int32_t verticesPerThreadBlock){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock+threadIdx.x;

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat+=blockDim.x){
		vertexId_t vpos=v_init+v_hat;
		if(vpos>=len){
			break;
		}
		vertexId_t v=verArray[vpos];

		(*cusSK)(custing,v,metadata);

	}
}


__global__ void allVerticesInArrayOneVertexPerTB(vertexId_t* verArray, length_t len,cuStinger* custing,void* metadata, cusSubKernel* cusSK, int32_t verticesPerThreadBlock)
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

__global__ void allVerticesInArrayOneVertexPerTB_LB(vertexId_t* verArray, length_t len,cuStinger* custing,void* metadata, cusSubKernel* cusSK,length_t* needles, length_t numNeedles)
{
	vertexId_t v_init=needles[blockIdx.x];
	vertexId_t v_diff=needles[blockIdx.x+1]-needles[blockIdx.x];

	// if(threadIdx.x==0 && blockIdx.x==(numNeedles-1))
	// 	printf("%d %d %d\n",blockIdx.x, v_init, v_diff);

	if(blockIdx.x==(numNeedles-1))
		v_diff=custing->nv-needles[blockIdx.x];

	for (vertexId_t v_hat=0; v_hat<(v_diff); v_hat++){
		vertexId_t vpos=v_init+v_hat;
		if(vpos>=len){
			break;
		}
		vertexId_t v=verArray[vpos];
		(*cusSK)(custing,v,metadata);
		// printf("\n");
		// if(threadIdx.x==0 && blockIdx.x==99)
		// 	printf("%d, ", v);
	}
}

__global__ void allVerticesInArray_LB_EXTRA(vertexId_t* verArray, length_t len,cuStinger* custing,void* metadata, cusSubKernel* cusSK,length_t* needles, length_t numNeedles)
{
	vertexId_t v_init=needles[blockIdx.x];
	vertexId_t v_diff=needles[blockIdx.x+1]-needles[blockIdx.x];
	if(blockIdx.x==(numNeedles-1))
		v_diff=custing->nv-needles[blockIdx.x];

	const int elePerThread = 8;

    // Specialize BlockRadixSort type for our thread block
    typedef BlockRadixSort<length_t, 32, elePerThread,vertexId_t> BlockRadixSortT;

    __shared__ typename BlockRadixSortT::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    length_t   thread_keys[elePerThread];
    vertexId_t thread_values[elePerThread];
    // Collectively sort the keys

    __shared__ length_t   sorted_keys[32*elePerThread];
    __shared__ vertexId_t sorted_values[32*elePerThread];

    vertexId_t v_hat=0;

    while(v_hat<v_diff){
    	vertexId_t v_hat2=v_hat+threadIdx.x;
    	for(int i=0; i<elePerThread; i++){
			vertexId_t vpos=v_init+v_hat2;
    		if(v_hat2>=v_diff){
    			thread_keys[i]   = -1;
    			thread_values[i] = 1000000000;
    		}
    		else{
    			vertexId_t temp  = verArray[vpos];
    			thread_keys[i]   = custing->dVD->used[temp];
    			thread_values[i] = temp;
    		}
    		v_hat2+=32;
    	}
    	__syncthreads();
	    BlockRadixSortT(temp_storage).SortDescending(thread_keys,thread_values);
    	__syncthreads();

	    int pos=threadIdx.x*elePerThread;
	    for(int i=0; i< elePerThread; i++){
		    sorted_keys[pos+i]     = thread_keys[i];
		    sorted_values[pos+i]   = thread_values[i];
	    }
	    __syncthreads();

	 //    if(blockIdx.x==99 && threadIdx.x==0){
	 //    		for(int i=0; i<128; i++){
	 //    	    	// printf("%d %d %d %d \n ", thread_keys[0],thread_keys[1],thread_keys[2],thread_keys[3]);
	 //    	    	// printf("%d, ", sorted_values[i]);
	 //    	    	// printf("%d %d\n ", sorted_keys[i],sorted_values[i]);
	 //    	    }
	 //    	    // printf("\n");
		// }

	    int break_early=-1;
		for (int u_hat=0; u_hat<32*elePerThread; u_hat++){
			if(sorted_keys[u_hat]==-1)
				break;
			if(sorted_keys[u_hat]<64){
				break_early=u_hat;
				break;
			}
			vertexId_t v=sorted_values[u_hat];
			(*cusSK)(custing,v,metadata);
			// if(blockIdx.x==99 && threadIdx.x==0)
			// 	printf("%d, ",sorted_values[u_hat]);
		}

	    __syncthreads();

				// vertexId_t v=sorted_values[u_hat];
				// (*cusSK)(custing,v,metadata);

		// __syncthreads();
		if(break_early!=-1){

			// for (int u_hat=break_early; u_hat<128; u_hat++){
			// 	if(sorted_keys[u_hat]==-1)
			// 		break;

			// 	vertexId_t v=sorted_values[u_hat];
			// 	(*cusSK)(custing,v,metadata);
			// 	// if(blockIdx.x==99 && threadIdx.x==0)
			// 	// 	printf("%d, ",sorted_values[u_hat]);
			// }


			for (int u_hat=break_early+threadIdx.x; u_hat<32*elePerThread; u_hat+=32){

				if(sorted_keys[u_hat]!=-1)
				{
				// if(blockIdx.x==99)
				// 	printf("%d, ", sorted_values[u_hat]);

					vertexId_t src=sorted_values[u_hat];

					connectedComponentsData* ccd = (connectedComponentsData*)metadata;

					length_t srcLen=custing->dVD->used[src];
					vertexId_t* adj_src=custing->dVD->adj[src]->dst;
					vertexId_t prev=ccd->prevState[src];

					// for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
					for(vertexId_t adj=0; adj<srcLen; adj++){
						vertexId_t dest = adj_src[adj];
						if (atomicMin(ccd->currState+src,ccd->prevState[dest]) < prev)
							atomicAdd(&(ccd->changeCurrIter),1);
					}

				}
				__syncthreads();
			}

		}

		// if(blockIdx.x==99 && threadIdx.x==0)
		// 	printf("\n");

	    v_hat+=32*elePerThread	;
    }


}




__device__ void workloadByAdjacency(cuStinger* custing,vertexId_t src, void* metadata){
	length_t* workload = (length_t*)metadata;
	workload[src]=custing->dVD->used[src];

}
__device__ cusSubKernel ptrWorkloadByAdjacency = workloadByAdjacency;

__global__ void createNeedles(length_t* dNeedles,length_t numNeedles, length_t* dPrefixArray, length_t prefixSize){
	if(threadIdx.x==0)
		dNeedles[blockIdx.x]=dPrefixArray[prefixSize-1]*((double)(blockIdx.x)/(double)numNeedles);
}


standard_context_t context(false);

void loadBalance(vertexId_t* verArray, length_t len,cuStinger custing,void* metadata, 
	length_t* dPartitionPoints, length_t* numPartitionPoints){
	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock;
	// standard_context_t context(false);

	numBlocks.x = ceil((float)len/(float)threads);
	if (numBlocks.x>64000)
		numBlocks.x=64000;
	verticesPerThreadBlock = ceil(float(len)/float(numBlocks.x-1));

	length_t* lbArray = (length_t*)allocDeviceArray(len+1,sizeof(length_t));
	length_t* dPrefixArray = (length_t*)allocDeviceArray(len+1,sizeof(length_t));

	cusSubKernel* dLoadBalance = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol(dLoadBalance, ptrWorkloadByAdjacency, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);

	allVerticesInArrayParallelVertexPerTB<<<numBlocks, threadsPerBlock>>>(verArray, len,custing.devicePtr(), 
										lbArray,dLoadBalance,verticesPerThreadBlock);

	scan(lbArray, len+1, dPrefixArray, context);

	// length_t* hPrefixArray = (length_t*)allocHostArray(len+1,sizeof(length_t));
	// copyArrayDeviceToHost(dPrefixArray,hPrefixArray,len+1, sizeof(length_t));
	// for(int v=0; v<=20; v++)
	// 	cout << tempHost[v] << ", " << hPrefixArray[v] << ", " << endl;

	// creating needles
	int numNeedles = *numPartitionPoints;//1000;
	// length_t* hNeedles = (length_t*)allocHostArray(numNeedles,sizeof(length_t));
	length_t* dNeedles = (length_t*)allocDeviceArray(numNeedles,sizeof(length_t));

	// length_t* dPositions = (length_t*)allocDeviceArray(numNeedles,sizeof(length_t));
	length_t* dPositions = dPartitionPoints;
	// length_t* hPositions = (length_t*)allocHostArray(numNeedles,sizeof(length_t));

	// for(int n=0; n<numNeedles; n++)
	// 	hNeedles[n]=hPrefixArray[len]*((double)n/(double)numNeedles);
	createNeedles<<<numNeedles,32>>>(dNeedles,numNeedles,dPrefixArray, len+1);

	// copyArrayHostToDevice(hNeedles,dNeedles,numNeedles, sizeof(length_t));
	sorted_search<bounds_lower>(dNeedles, numNeedles, dPrefixArray, len, dPositions, less_t<int>(), context);

	// copyArrayDeviceToHost(dPositions,hPositions,numNeedles, sizeof(length_t));
	// for(int n=0; n<numNeedles; n++)
	// 	cout << n << "  " << hPositions[n] <<" "  << endl;

	// freeHostArray(hPositions);
	// freeDeviceArray(dPositions);
	freeDeviceArray(dNeedles);
	// freeHostArray(hNeedles);

	// freeHostArray(hPrefixArray);
	freeDeviceArray(dPrefixArray);
	freeDeviceArray(lbArray);
	freeDeviceArray(dLoadBalance);

	// Compiles correctly
	// sorted_search<bounds_lower>(lbArray, 100, bs.data(), 1000, bs.data(), less_t<int>(), context);
	// void sorted_search(needles_it needles, int num_needles, haystack_it haystack,
 //  		int num_haystack, indices_it indices, comp_it comp, context_t& context)

}



/*
	########  ########  ######          ######## ##     ## ##    ##  ######  ########  #######  ########   ######
	##     ## ##       ##    ##         ##       ##     ## ###   ## ##    ##    ##    ##     ## ##     ## ##    ##
	##     ## ##       ##               ##       ##     ## ####  ## ##          ##    ##     ## ##     ## ##
	########  ######    ######  ####### ######   ##     ## ## ## ## ##          ##    ##     ## ########   ######
	##     ## ##             ##         ##       ##     ## ##  #### ##          ##    ##     ## ##   ##         ##
	##     ## ##       ##    ##         ##       ##     ## ##   ### ##    ##    ##    ##     ## ##    ##  ##    ##
	########  ##        ######          ##        #######  ##    ##  ######     ##     #######  ##     ##  ######
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

__device__ void bfsExpandFrontierMacro(cuStinger* custing,vertexId_t src, void* metadata){
	bfsData* bd = (bfsData*)metadata;
	vertexId_t nextLevel=bd->currLevel+1;

	CUSTINGER_FOR_ALL_EDGES_OF_VERTEX_PAR_THREAD_BLOCK_BEGIN(custing,src) 
		vertexId_t prev = atomicCAS(bd->level+CUSTINGER_EDGE_DEST,INT32_MAX,nextLevel);
		if(prev==INT32_MAX){
			length_t prevPos = atomicAdd(&(bd->queueEnd),1);
			bd->queue[prevPos] = CUSTINGER_EDGE_DEST;
		}
	CUSTINGER_FOR_ALL_EDGES_OF_VERTEX_PAR_THREAD_BLOCK_END()
}
__device__ cusSubKernel ptrBFSExpandFrontierMacro = bfsExpandFrontierMacro;


__device__ void setLevelInfinity(cuStinger* custing,vertexId_t src, void* metadata){
	bfsData* bd = (bfsData*)metadata;
	bd->level[src]=INT32_MAX;
}
__device__ cusSubKernel ptrSetLevelInfinity = setLevelInfinity;



/*
	########  ########  ######          ##     ##    ###    #### ##    ##
	##     ## ##       ##    ##         ###   ###   ## ##    ##  ###   ##
	##     ## ##       ##               #### ####  ##   ##   ##  ####  ##
	########  ######    ######  ####### ## ### ## ##     ##  ##  ## ## ##
	##     ## ##             ##         ##     ## #########  ##  ##  ####
	##     ## ##       ##    ##         ##     ## ##     ##  ##  ##   ###
	########  ##        ######          ##     ## ##     ## #### ##    ##
*/

void bfsMain(cuStinger& custing, void* func_meta_data)
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
	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceBfsData,dSetInfinity2,verticesPerThreadBlock);
	freeDeviceArray(dSetInfinity2);

	vertexId_t root=2; length_t level=0;
	copyArrayHostToDevice(&root,hostBfsData.queue,1,sizeof(vertexId_t));
	copyArrayHostToDevice(&level,hostBfsData.level+root,1,sizeof(length_t));

	cusSubKernel* dTraverseEdges = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	// cudaMemcpyFromSymbol( dTraverseEdges, ptrBFSExpandFrontierMacro, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	cudaMemcpyFromSymbol( dTraverseEdges, ptrBFSExpandFrontier, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);

	length_t prevEnd=1;
	while(hostBfsData.queueEnd-hostBfsData.queueCurr>0){
		allVerticesInArrayOneVertexPerTB<<<numBlocks, threadsPerBlock>>>(hostBfsData.queue+hostBfsData.queueCurr,
										hostBfsData.queueEnd-hostBfsData.queueCurr,custing.devicePtr(),
										deviceBfsData,dTraverseEdges,verticesPerThreadBlock);
		copyArrayDeviceToHost(deviceBfsData,&hostBfsData,1, sizeof(bfsData));

		hostBfsData.queueCurr=prevEnd;
		prevEnd = hostBfsData.queueEnd;
		hostBfsData.currLevel++;
		copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));
	}

	freeDeviceArray(dTraverseEdges);
	cout << "The queue end  :" << hostBfsData.queueEnd << endl;

	float totalBFSTime = end_clock(ce_start, ce_stop);
	cout << "Total time for the BFS : " << totalBFSTime << endl; 

	freeDeviceArray(deviceBfsData);
	freeDeviceArray(hostBfsData.queue);
	freeDeviceArray(hostBfsData.level);
}


/*
	 ######   #######  ##    ## ##    ## ########  ######  ########          ######   #######  ##     ## ########   #######  ##    ## ######## ##    ## ########  ######          ######## ##     ## ##    ##  ######  ########  #######  ########   ######
	##    ## ##     ## ###   ## ###   ## ##       ##    ##    ##            ##    ## ##     ## ###   ### ##     ## ##     ## ###   ## ##       ###   ##    ##    ##    ##         ##       ##     ## ###   ## ##    ##    ##    ##     ## ##     ## ##    ##
	##       ##     ## ####  ## ####  ## ##       ##          ##            ##       ##     ## #### #### ##     ## ##     ## ####  ## ##       ####  ##    ##    ##               ##       ##     ## ####  ## ##          ##    ##     ## ##     ## ##
	##       ##     ## ## ## ## ## ## ## ######   ##          ##    ####### ##       ##     ## ## ### ## ########  ##     ## ## ## ## ######   ## ## ##    ##     ######  ####### ######   ##     ## ## ## ## ##          ##    ##     ## ########   ######
	##       ##     ## ##  #### ##  #### ##       ##          ##            ##       ##     ## ##     ## ##        ##     ## ##  #### ##       ##  ####    ##          ##         ##       ##     ## ##  #### ##          ##    ##     ## ##   ##         ##
	##    ## ##     ## ##   ### ##   ### ##       ##    ##    ##            ##    ## ##     ## ##     ## ##        ##     ## ##   ### ##       ##   ###    ##    ##    ##         ##       ##     ## ##   ### ##    ##    ##    ##     ## ##    ##  ##    ##
	 ######   #######  ##    ## ##    ## ########  ######     ##             ######   #######  ##     ## ##         #######  ##    ## ######## ##    ##    ##     ######          ##        #######  ##    ##  ######     ##     #######  ##     ##  ######
*/


__device__ void connectedComponentInit(cuStinger* custing,vertexId_t src, void* metadata){
	connectedComponentsData* ccd = (connectedComponentsData*)metadata;
	ccd->currState[src]=src;
}
__device__ cusSubKernel ptrConnectedComponentInit = connectedComponentInit;


__device__ void connectedComponentsSwap(cuStinger* custing,vertexId_t src, void* metadata){
	connectedComponentsData* ccd = (connectedComponentsData*)metadata;

	length_t srcLen=custing->dVD->used[src];
	vertexId_t* adj_src=custing->dVD->adj[src]->dst;
	vertexId_t prev=ccd->prevState[src];
	for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
		vertexId_t dest = adj_src[adj];
		if (atomicMin(ccd->currState+src,ccd->prevState[dest]) < prev)
			atomicAdd(&(ccd->changeCurrIter),1);
	}
}
__device__ cusSubKernel ptrConnectedComponentSwap = connectedComponentsSwap;

__device__ void connectedComponentsSwapLocal(cuStinger* custing,vertexId_t src, void* metadata){
	connectedComponentsData* ccd = (connectedComponentsData*)metadata;

	length_t srcLen=custing->dVD->used[src];
	vertexId_t* adj_src=custing->dVD->adj[src]->dst;
	vertexId_t prev=ccd->currState[src];
	for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
		vertexId_t dest = adj_src[adj];
		if (atomicMin(ccd->currState+src,ccd->currState[dest]) < prev)
			atomicAdd(&(ccd->changeCurrIter),1);
	}
}
__device__ cusSubKernel ptrConnectedComponentSwapLocal = connectedComponentsSwapLocal;


__device__ void connectedComponentShortcut(cuStinger* custing,vertexId_t src, void* metadata){
	connectedComponentsData* ccd = (connectedComponentsData*)metadata;
	ccd->currState[src] = ccd->currState[ccd->currState[src]]; 
}
__device__ cusSubKernel ptrConnectedComponentShortcut = connectedComponentShortcut;

__device__ void connectedComponentCount(cuStinger* custing,vertexId_t src, void* metadata){
	connectedComponentsData* ccd = (connectedComponentsData*)metadata;
	if(ccd->currState[src] == src){
		atomicAdd(&(ccd->connectedComponentCount),1);	
	}
}
__device__ cusSubKernel ptrConnectedComponentCount = connectedComponentCount;



__global__ void allVerticesInGraphOneVertexPerThreadBlockBAA(cuStinger* custing,void* metadata, cusSubKernel* cusSK, int32_t verticesPerThreadBlock){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock;
	length_t nv = custing->getMaxNV();
	connectedComponentsData* ccd = (connectedComponentsData*)metadata;

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		vertexId_t src=v_init+v_hat;
		if(src>=nv){
			break;
		}
		length_t srcLen=custing->dVD->used[src];
		vertexId_t* adj_src=custing->dVD->adj[src]->dst;
		vertexId_t prev=ccd->currState[src];
		for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
			vertexId_t dest = adj_src[adj];
			if (atomicMin(ccd->currState+src,ccd->currState[dest]) < prev)
				atomicAdd(&(ccd->changeCurrIter),1);
		}
	}
}


/*
	 ######   #######  ##    ## ##    ## ########  ######  ######## ######## ########           ######   #######  ##     ## ########   #######  ##    ## ######## ##    ## ########  ######          ##     ##    ###    #### ##    ##
	##    ## ##     ## ###   ## ###   ## ##       ##    ##    ##    ##       ##     ##         ##    ## ##     ## ###   ### ##     ## ##     ## ###   ## ##       ###   ##    ##    ##    ##         ###   ###   ## ##    ##  ###   ##
	##       ##     ## ####  ## ####  ## ##       ##          ##    ##       ##     ##         ##       ##     ## #### #### ##     ## ##     ## ####  ## ##       ####  ##    ##    ##               #### ####  ##   ##   ##  ####  ##
	##       ##     ## ## ## ## ## ## ## ######   ##          ##    ######   ##     ## ####### ##       ##     ## ## ### ## ########  ##     ## ## ## ## ######   ## ## ##    ##     ######  ####### ## ### ## ##     ##  ##  ## ## ##
	##       ##     ## ##  #### ##  #### ##       ##          ##    ##       ##     ##         ##       ##     ## ##     ## ##        ##     ## ##  #### ##       ##  ####    ##          ##         ##     ## #########  ##  ##  ####
	##    ## ##     ## ##   ### ##   ### ##       ##    ##    ##    ##       ##     ##         ##    ## ##     ## ##     ## ##        ##     ## ##   ### ##       ##   ###    ##    ##    ##         ##     ## ##     ##  ##  ##   ###
	 ######   #######  ##    ## ##    ## ########  ######     ##    ######## ########           ######   #######  ##     ## ##         #######  ##    ## ######## ##    ##    ##     ######          ##     ## ##     ## #### ##    ##
*/




void connectComponentsMain(cuStinger& custing, void* func_meta_data)
{
	cudaEvent_t ce_start,ce_stop;	
	start_clock(ce_start, ce_stop);

	connectedComponentsData hostCCData;
	hostCCData.queue = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.currState = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.prevState = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.changeCurrIter=1;
	hostCCData.connectedComponentCount=0;
	connectedComponentsData* deviceCCData = (connectedComponentsData*)allocDeviceArray(1, sizeof(connectedComponentsData));
	copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));


	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock=128;

	numBlocks.x = ceil((float)custing.nv/(float)verticesPerThreadBlock);

	cusSubKernel* dSetInit = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dSetInit, ptrConnectedComponentInit, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dSetInit,verticesPerThreadBlock);
	freeDeviceArray(dSetInit);

	cusSubKernel* dswap = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dswap, ptrConnectedComponentSwap, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	cusSubKernel* dshortcut = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dshortcut, ptrConnectedComponentShortcut, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	cusSubKernel* dcount = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dcount, ptrConnectedComponentCount, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);

	copyArrayDeviceToDevice(hostCCData.currState,hostCCData.prevState,custing.nv,sizeof(vertexId_t));
	
	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dcount,verticesPerThreadBlock);
	copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(connectedComponentsData));

	int iteration = 0;
	while(hostCCData.changeCurrIter){
		// cout << "The number of connected-compoents  : " << hostCCData.connectedComponentCount << endl;
		// cout << "The number of swaps  : " << hostCCData.connectedComponentCount << endl;

		hostCCData.changeCurrIter=0;
		hostCCData.connectedComponentCount=0;

		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));
		// copyArrayDeviceToDevice(hostCCData.prevState,hostCCData.currState,custing.nv,sizeof(vertexId_t));
		allVerticesInGraphOneVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dswap,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dshortcut,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dshortcut,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dcount,verticesPerThreadBlock);

		copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(connectedComponentsData));

		vertexId_t* temp     = hostCCData.prevState;
		hostCCData.prevState = hostCCData.currState;
		hostCCData.currState = temp;
		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));
		iteration++;
	}
	freeDeviceArray(dswap);
	freeDeviceArray(dshortcut);
	freeDeviceArray(dcount);

	float totalBFSTime = end_clock(ce_start, ce_stop);

	cout << "The number of iterations           : " << iteration << endl;
	cout << "The number of connected-compoents  : " << hostCCData.connectedComponentCount << endl;
	cout << "Total time for connected-compoents : " << totalBFSTime << endl; 

	freeDeviceArray(deviceCCData);
	freeDeviceArray(hostCCData.queue);
	freeDeviceArray(hostCCData.currState);
	freeDeviceArray(hostCCData.prevState);
}

void connectComponentsMainLocal(cuStinger& custing, void* func_meta_data)
{
	cudaEvent_t ce_start,ce_stop;	
	start_clock(ce_start, ce_stop);

	connectedComponentsData hostCCData;
	hostCCData.queue = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.currState = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.changeCurrIter=1;
	hostCCData.connectedComponentCount=0;
	connectedComponentsData* deviceCCData = (connectedComponentsData*)allocDeviceArray(1, sizeof(connectedComponentsData));
	copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));

	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock=128;

	numBlocks.x = ceil((float)custing.nv/(float)verticesPerThreadBlock);

	cusSubKernel* dSetInit = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dSetInit, ptrConnectedComponentInit, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dSetInit,verticesPerThreadBlock);
	freeDeviceArray(dSetInit);

	cusSubKernel* dswapLocal = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dswapLocal, ptrConnectedComponentSwapLocal, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);

	cusSubKernel* dshortcut = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dshortcut, ptrConnectedComponentShortcut, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	cusSubKernel* dcount = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dcount, ptrConnectedComponentCount, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);	

	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dcount,verticesPerThreadBlock);
	copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(connectedComponentsData));


	int iteration = 0;
	while(hostCCData.changeCurrIter){
		// cout << "The number of connected-compoents  : " << hostCCData.connectedComponentCount << endl;
		// cout << "The number of swaps  : " << hostCCData.connectedComponentCount << endl;

		hostCCData.changeCurrIter=0;
		hostCCData.connectedComponentCount=0;

		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));

		allVerticesInGraphOneVertexPerThreadBlockBAA<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dswapLocal,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dshortcut,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dshortcut,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dcount,verticesPerThreadBlock);
		copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(connectedComponentsData));

		iteration++;
	}
	freeDeviceArray(dswapLocal);
	freeDeviceArray(dshortcut);
	freeDeviceArray(dcount);

	float totalBFSTime = end_clock(ce_start, ce_stop);

	cout << "The number of iterations           : " << iteration << endl;
	cout << "The number of connected-compoents  : " << hostCCData.connectedComponentCount << endl;
	cout << "Total time for connected-compoents : " << totalBFSTime << endl; 

	freeDeviceArray(deviceCCData);
	freeDeviceArray(hostCCData.queue);
	freeDeviceArray(hostCCData.currState);
}




void oneMoreMain(cuStinger& custing, void* func_meta_data){
	vertexId_t* h_allVerts = (vertexId_t*)allocHostArray(custing.nv,sizeof(vertexId_t));
	vertexId_t* d_allVerts = (vertexId_t*)allocDeviceArray(custing.nv,sizeof(vertexId_t));

	cudaEvent_t ce_start,ce_stop;	
	start_clock(ce_start, ce_stop);

	length_t numPartitionPoints=1000;
	// length_t* dNeedles = (length_t*)allocDeviceArray(numPartitionPoints,sizeof(length_t));
	length_t* dPositions = (length_t*)allocDeviceArray(numPartitionPoints,sizeof(length_t));

	for(int v=0; v<custing.nv; v++)
		h_allVerts[v]=v;
	copyArrayHostToDevice(h_allVerts,d_allVerts,custing.nv,sizeof(vertexId_t));

	// loadBalance(d_allVerts, custing.nv,custing,NULL,NULL,NULL);
	loadBalance(d_allVerts, custing.nv,custing,NULL,dPositions,&numPartitionPoints);

	// float totalBFSTime = end_clock(ce_start, ce_stop);

	// cout << "The number of iterations           : " << iteration << endl;
	// cout << "The number of connected-compoents  : " << hostCCData.connectedComponentCount << endl;
	// cout << "Total time for load-balancing : " << totalBFSTime << endl; 

	// cudaEvent_t ce_start,ce_stop;	
	// start_clock(ce_start, ce_stop);

	connectedComponentsData hostCCData;
	hostCCData.queue = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.currState = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.prevState = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	hostCCData.changeCurrIter=1;
	hostCCData.connectedComponentCount=0;
	connectedComponentsData* deviceCCData = (connectedComponentsData*)allocDeviceArray(1, sizeof(connectedComponentsData));
	copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));


	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock=128;

	numBlocks.x = ceil((float)custing.nv/(float)verticesPerThreadBlock);

	cusSubKernel* dSetInit = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dSetInit, ptrConnectedComponentInit, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dSetInit,verticesPerThreadBlock);
	freeDeviceArray(dSetInit);

	cusSubKernel* dswap = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dswap, ptrConnectedComponentSwap, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	cusSubKernel* dshortcut = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dshortcut, ptrConnectedComponentShortcut, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
	cusSubKernel* dcount = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
	cudaMemcpyFromSymbol( dcount, ptrConnectedComponentCount, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);

	copyArrayDeviceToDevice(hostCCData.currState,hostCCData.prevState,custing.nv,sizeof(vertexId_t));
	
	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dcount,verticesPerThreadBlock);
	copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(connectedComponentsData));

	int iteration = 0;
	while(hostCCData.changeCurrIter){
		// cout << "The number of connected-compoents  : " << hostCCData.connectedComponentCount << endl;
		// cout << "The number of swaps  : " << hostCCData.connectedComponentCount << endl;

		hostCCData.changeCurrIter=0;
		hostCCData.connectedComponentCount=0;

		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));
		// allVerticesInArrayOneVertexPerTB_LB<<<numPartitionPoints,32>>>(d_allVerts,custing.nv,custing.devicePtr(),deviceCCData,dswap,dPositions,numPartitionPoints);
		allVerticesInArray_LB_EXTRA<<<numPartitionPoints,32>>>(d_allVerts,custing.nv,custing.devicePtr(),deviceCCData,dswap,dPositions,numPartitionPoints);
		// allVerticesInGraphOneVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dswap,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dshortcut,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dshortcut,verticesPerThreadBlock);
		allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceCCData,dcount,verticesPerThreadBlock);

		copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(connectedComponentsData));

		vertexId_t* temp     = hostCCData.prevState;
		hostCCData.prevState = hostCCData.currState;
		hostCCData.currState = temp;
		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(connectedComponentsData));
		// printf("Iteration %d",iteration);

		iteration++;
	}
	freeDeviceArray(dswap);
	freeDeviceArray(dshortcut);
	freeDeviceArray(dcount);

	float totalBFSTime = end_clock(ce_start, ce_stop);

	cout << "The number of iterations           : " << iteration << endl;
	cout << "The number of connected-compoents  : " << hostCCData.connectedComponentCount << endl;
	cout << "Total time for connected-compoents : " << totalBFSTime << endl; 

	freeDeviceArray(deviceCCData);
	freeDeviceArray(hostCCData.queue);
	freeDeviceArray(hostCCData.currState);
	freeDeviceArray(hostCCData.prevState);

	freeDeviceArray(dPositions);

	freeHostArray(h_allVerts);
	freeDeviceArray(d_allVerts);


}
