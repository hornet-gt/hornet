
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.hpp"

using namespace std;


// void initializeCuStinger(cuStingerConfig);


__global__ void devMakeGPUStinger(int32_t* d_off, int32_t* d_adj,
	int verticesPerThreadBlock,cuStinger* custing)
{
	int32_t** d_cuadj = custing->d_adj;
	length_t* d_utilized = custing->d_utilized;

	int32_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		int32_t v=v_init+v_hat;
		if(v>=custing->nv)
			break;
		for(int32_t e=threadIdx.x; e<d_utilized[v]; e+=blockDim.x){
			d_cuadj[v][e]=d_adj[d_off[v]+e];
		}
	}
}


void cuStinger::internalCSRcuStinger(length_t* h_off, vertexId_t* h_adj, int ne){
	length_t* d_off = (length_t*)allocDeviceArray(nv+1,sizeof(int32_t));
	vertexId_t* d_adj = (length_t*)allocDeviceArray(ne,sizeof(int32_t));
	copyArrayHostToDevice(h_off,d_off,nv,sizeof(length_t));
	copyArrayHostToDevice(h_adj,d_adj,ne,sizeof(vertexId_t));

	dim3 numBlocks(1, 1);
	int32_t threads=64;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)nv/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	

	int32_t verticesPerThreadBlock = ceil(float(nv)/float(numBlocks.x-1));

	devMakeGPUStinger<<<numBlocks,threadsPerBlock>>>(d_off,d_adj,verticesPerThreadBlock, d_cuStinger);

	freeDeviceArray(d_adj);	
	freeDeviceArray(d_off);
}



#define SUM_BLOCK_SIZE 512
__global__ void total(length_t * input, length_t * output, length_t len) {
    __shared__ length_t partialSum[2 * SUM_BLOCK_SIZE];
    //Load a segment of the input vector into shared memory
    length_t tid = threadIdx.x, start = 2 * blockIdx.x * SUM_BLOCK_SIZE;
    if (start + tid < len)
       partialSum[tid] = input[start + tid];
    else
       partialSum[tid] = 0;

    if (start + SUM_BLOCK_SIZE + tid < len)
       partialSum[SUM_BLOCK_SIZE + tid] = input[start + SUM_BLOCK_SIZE + tid];
    else
       partialSum[SUM_BLOCK_SIZE + tid] = 0;

    //Traverse the reduction tree
    for (int stride = SUM_BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (tid < stride)
          partialSum[tid] += partialSum[tid+stride];
    }
    //Write the computed sum of the block to the output vector at the correct index
    if (tid == 0)
       output[blockIdx.x] = partialSum[0];
}


length_t cuStinger::sumDeviceArray(length_t* arr){
	length_t numOutputElements = nv / (SUM_BLOCK_SIZE<<1);
    if (nv % (SUM_BLOCK_SIZE<<1)) {
        numOutputElements++;
    }

	length_t* d_out = (length_t*)allocDeviceArray(nv, sizeof(length_t*));

	total<<<numOutputElements,SUM_BLOCK_SIZE>>>(d_utilized,d_out,nv);

	length_t* h_out = (int32_t*)allocHostArray(nv, sizeof(length_t*));
	
	length_t sum=0;
	copyArrayDeviceToHost(d_out, h_out, nv, sizeof(length_t));
	for(int i=0; i<numOutputElements; i++){
		 // cout << h_out[i] << ", ";
		sum+=h_out[i];
	}
	freeHostArray(h_out);
	freeDeviceArray(d_out);	
	return sum;
}



__global__ void deviceCopyMultipleAdjacencies(cuStinger* custing, vertexId_t** d_newadj, 
	vertexId_t* requireUpdates, length_t requireCount ,length_t verticesPerThreadBlock)
{
	int32_t** d_cuadj = custing->d_adj;
	length_t* d_utilized = custing->d_utilized;

	length_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		if((v_init+v_hat)>=requireCount)
			break;
		vertexId_t v=requireUpdates[v_init+v_hat];

		for(length_t e=threadIdx.x; e<d_utilized[v]; e+=blockDim.x){
			d_newadj[v][e] = d_cuadj[v][e];
		}
	}
}

void cuStinger::copyMultipleAdjacencies(vertexId_t** d_newadj, 
	vertexId_t* requireUpdates, length_t requireCount){

	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)requireCount);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	int32_t verticesPerThreadBlock;
	if(numBlocks.x == requireCount)
		verticesPerThreadBlock=1;
	else
		verticesPerThreadBlock = ceil(float(requireCount)/float(numBlocks.x-1));

	cout << "### " << requireCount << " , " <<  numBlocks.x << " , " << verticesPerThreadBlock << " ###"  << endl; 

	deviceCopyMultipleAdjacencies<<<numBlocks,threadsPerBlock>>>(d_cuStinger,
		d_newadj, requireUpdates, requireCount, verticesPerThreadBlock);
	checkLastCudaError("Error in the first update sweep");
}



