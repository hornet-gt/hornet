
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



void copyArrayHostToDevice(void* hostSrc, void* devDst, int32_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(devDst,hostSrc,elements*eleSize,cudaMemcpyHostToDevice));
}

void copyArrayDeviceToHost(void* devSrc, void* hostDst, int32_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(hostDst,devSrc,elements*eleSize,cudaMemcpyDeviceToHost));
}

void copyArrayDeviceToDevice(void* devSrc, void* devDst, int32_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(hostDst,devSrc,elements*eleSize,cudaMemcpyDeviceToDevice));
}


