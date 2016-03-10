
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.h"

using namespace std;


void* allocHostArray(int32_t elements,int32_t eleSize){
	if (elements==0 || eleSize==0)
		return NULL;
	return malloc(eleSize*elements);
}

void* allocDeviceArray(int32_t elements,int32_t eleSize){
	int32_t* ptr=NULL;
	if (elements==0 || eleSize==0)
		return NULL;
	checkCudaErrors (cudaMalloc((void **)&ptr,eleSize*elements));
	return ptr;
}

void freeHostArray(void* array){
	free(array);
}

void freeDeviceArray(void* array){
	checkCudaErrors(cudaFree(array));
}

int32_t elementsPerVertex(int32_t elements){
	int32_t eleCount = elements;
	if(eleCount==0)
		eleCount=1;
	else if(eleCount < 5)
		eleCount*=2;
	else
		eleCount*=1.5;
	return eleCount;
}

void copyArrayHostToDevice(void* hostSrc, void* devDst, int32_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(devDst,hostSrc,elements*eleSize,cudaMemcpyHostToDevice));
}

void copyArrayDeviceToHost(void* devSrc, void* hostDst, int32_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(hostDst,devSrc,elements*eleSize,cudaMemcpyDeviceToHost));
	// cout << "D to H error : "<<  cudaGetErrorString(code) << endl;
}

__global__ void devMakeGPUStinger(int32_t nv,int32_t ne,int32_t* d_off, int32_t* d_adj,
	int verticesPerThreadBlock,int32_t** d_adjArray, int32_t* d_adjSizeUsed){
	int32_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		int32_t v=v_init+v_hat;
		if(v>=nv)
			break;

		for(int32_t e=threadIdx.x; e<d_adjSizeUsed[v]; e+=blockDim.x){
			d_adjArray[v][e]=d_adj[d_off[v]+e];
		}

	}
}

void hostMakeGPUStinger(int32_t nv,int32_t ne,int32_t* h_off, int32_t* h_adj,
	int32_tPtr* d_adjArray,int32_t* d_adjSizeUsed,int32_t* d_adjSizeMax){

	int32_t* d_off = (int32_t*)allocDeviceArray(nv+1,sizeof(int32_t));
	int32_t* d_adj = (int32_t*)allocDeviceArray(ne,sizeof(int32_t));
	copyArrayHostToDevice(h_off,d_off,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_adj,d_adj,nv,sizeof(int32_t));


	dim3 numBlocks(1, 1);
	int32_t threads=64;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)nv/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	

	int32_t verticesPerThreadBlock = ceil(float(nv)/float(numBlocks.x-1));


	devMakeGPUStinger<<<numBlocks,threadsPerBlock>>>(nv,ne,d_off,d_adj,verticesPerThreadBlock,d_adjArray, d_adjSizeUsed);

	freeDeviceArray(d_adj);	
	freeDeviceArray(d_off);
}


void allocGPUMemory(int32_t nv,int32_t ne,int32_t* off, int32_t* adj,
	int32_tPtrPtr* d_adjArray,int32_t** d_adjSizeUsed,int32_t** d_adjSizeMax)
{	
	int32_tPtrPtr d_temp = (int32_t**)allocDeviceArray(nv,sizeof(int32_t*));
	*d_adjArray = d_temp;

	*d_adjSizeUsed = (int32_t*)allocDeviceArray(nv,sizeof(int32_t));
	*d_adjSizeMax =  (int32_t*)allocDeviceArray(nv,sizeof(int32_t));

	int32_tPtr* h_arrayPtr =  (int32_tPtr*)allocHostArray(nv,sizeof(int32_t*));
	int32_t* h_sizeArrayUsed =  (int32_t*)allocHostArray(nv,sizeof(int32_t));
	int32_t* h_sizeArrayMax =  (int32_t*)allocHostArray(nv,sizeof(int32_t));

	for(int v=0; v<nv; v++){
		h_sizeArrayUsed[v]=off[v+1]-off[v];
		h_sizeArrayMax[v] = elementsPerVertex(h_sizeArrayUsed[v]);
		h_arrayPtr[v] =  (int32_t*)allocDeviceArray(h_sizeArrayMax[v], sizeof(int32_t));
	}
	copyArrayHostToDevice(h_sizeArrayUsed,*d_adjSizeUsed,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_sizeArrayMax,*d_adjSizeMax,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_arrayPtr,*d_adjArray,nv,sizeof(int32_t*));

	freeHostArray(h_arrayPtr);
	freeHostArray(h_sizeArrayUsed);
	freeHostArray(h_sizeArrayMax);
}




