
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.h"
#include "cuStinger.hpp"

using namespace std;

__global__ void devMakeGPUStinger(int32_t* d_off, int32_t* d_adj,
	int verticesPerThreadBlock,cuStinger* custing)
{
	int32_t** d_cuadj = custing->d_adj;
	int32_t* d_utilized = custing->d_utilized;

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


void cuStinger::initcuStinger(int32_t* h_off, int32_t* h_adj){
	int32_t* d_off = (int32_t*)allocDeviceArray(nv+1,sizeof(int32_t));
	int32_t* d_adj = (int32_t*)allocDeviceArray(ne,sizeof(int32_t));
	copyArrayHostToDevice(h_off,d_off,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_adj,d_adj,ne,sizeof(int32_t));

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


template <int32_t blockSize>
__global__ void reduce(int32_t *g_idata, int32_t *g_odata, int32_t n)
{
	extern __shared__ int32_t sdata[];
	int32_t tid = threadIdx.x;
	int32_t i = blockIdx.x*(blockSize*2) + tid;
	int32_t gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


#define BLOCK_SIZE 512
__global__ void total(int32_t * input, int32_t * output, int32_t len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ int32_t partialSum[2 * BLOCK_SIZE];
    int32_t t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}


int32_t cuStinger::sumDeviceArray(int32_t* arr){
	int32_t numOutputElements = nv / (BLOCK_SIZE<<1);
    if (nv % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }

	int32_t* d_out = (int32_t*)allocDeviceArray(nv, sizeof(int32_t*));

	total<<<numOutputElements,512>>>(d_utilized,d_out,nv);

	int32_t* h_out = (int32_t*)allocHostArray(nv, sizeof(int32_t*));
	
	int32_t sum=0;
	copyArrayDeviceToHost(d_out, h_out, nv, sizeof(int32_t));
	for(int i=0; i<numOutputElements; i++){
		 // cout << h_out[i] << ", ";
		sum+=h_out[i];
	}
	freeHostArray(h_out);
	freeDeviceArray(d_out);	
	return sum;
}



__global__ void deviceCopyMultipleAdjacencies(cuStinger* custing, int32_t** d_newadj, 
	int32_t* requireUpdates, int32_t requireCount ,int32_t verticesPerThreadBlock)
{
	int32_t** d_cuadj = custing->d_adj;
	int32_t* d_utilized = custing->d_utilized;

	int32_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		int32_t v= requireUpdates[v_init+v_hat];
		if(v>=requireCount)
			break;
		for(int32_t e=threadIdx.x; e<d_utilized[v]; e+=blockDim.x){
			d_newadj[v][e] = d_cuadj[v][e];
			// d_cuadj[v][e] = d_cuadj[v][e];
		}
	}
}

void cuStinger::copyMultipleAdjacencies(int32_t** d_newadj, 
	int32_t* requireUpdates, int32_t requireCount){

	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)requireCount);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	int32_t verticesPerThreadBlock = ceil(float(requireCount)/float(numBlocks.x-1));

	cout << "### " << requireCount << " , " <<  numBlocks.x << " , " << verticesPerThreadBlock << " ###"  << endl; 

	deviceCopyMultipleAdjacencies<<<numBlocks,threadsPerBlock>>>(d_cuStinger,
		d_newadj, requireUpdates, requireCount, verticesPerThreadBlock);
	checkLastCudaError("Error in the first update sweep");
}



