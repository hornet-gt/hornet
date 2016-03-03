
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.h"

using namespace std;


// __device__ void  getSizes(int32_t* offSetArray, int32_t sizeArray, int32_t nv){
// 	int32_t pos = threadIdx.x + blockIdx.x*blockDim.x;

// 	if(pos<nv)
// 		sizeArray[pos]=offSetArray[pos+1]-offSetArray[pos];
// }

void* allocHostArray(int32_t elements,int32_t eleSize){
	if (elements==0 || eleSize==0)
		return NULL;
	return malloc(eleSize*elements);
}

void* allocDeviceArray(int32_t elements,int32_t eleSize){
	int32_t* ptr=NULL;
	if (elements==0 || eleSize==0)
		return NULL;
	cudaMalloc((void **)&ptr,eleSize*elements);
	return ptr;
}

void freeHostArray(void* array){
	free(array);
}

void freeDeviceArray(void* array){
	cudaFree(array);
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
	cudaMemcpy(hostSrc,devDst,elements*eleSize,cudaMemcpyHostToDevice);
}

void copyArrayDeviceToHost(void* devSrc, void* hostDst, int32_t elements, int32_t eleSize){
	cudaMemcpy(devSrc,hostDst,elements*eleSize,cudaMemcpyDeviceToHost);
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
	copyArrayHostToDevice(h_sizeArrayUsed,d_adjSizeUsed,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_sizeArrayMax,d_adjSizeMax,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_arrayPtr,*d_adjArray,nv,sizeof(int32_t*));

	freeHostArray(h_arrayPtr);
	freeHostArray(h_sizeArrayUsed);
	freeHostArray(h_sizeArrayMax);
}

__global__ void devMakeGPUStinger(int32_t nv,int32_t ne,int32_t* off, int32_t* adj,
	int32_t** d_adjArray, int32_t* d_adjSizeUsed){
	int32_t vertex=blockIdx.x;
	for(int32_t e=threadIdx.x; e<d_adjSizeUsed[vertex]; e+=32){

	}
}

void hostMakeGPUStinger(int32_t nv,int32_t ne,int32_t* off, int32_t* adj,
	thrust::device_vector<int32_t> *d_adjArray,thrust::device_vector<int32_t> d_adjSizeUsed){

	thrust::device_vector<int32_t> d_off (off,off+nv);
	thrust::device_vector<int32_t> d_adj (adj,adj+ne);

	// int32_t* temp = thrust::raw_pointer_cast(&d_off[0]);

	// devMakeGPUStinger<<<nv, 32>>>(nv,ne,NULL,NULL,NULL,NULL);//thrust::raw_pointer_cast(&d_adjSizeUsed[0]));

	devMakeGPUStinger<<<nv, 32>>>(nv,ne,thrust::raw_pointer_cast(&d_off[0]),
		thrust::raw_pointer_cast(&d_adj[0]),NULL,thrust::raw_pointer_cast(&d_adjSizeUsed[0]));
}

