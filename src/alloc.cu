
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdio.h>
#include <string.h>

#include "utils.hpp"

using namespace std;


void* allocHostArray(length_t elements,int32_t eleSize){
	if (elements==0 || eleSize==0)
		return NULL;
	return malloc(eleSize*elements);
}

void* allocDeviceArray(length_t elements,int32_t eleSize){
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


void copyArrayHostToHost(void* hostSrc,  void* hostDst, length_t elements, int32_t eleSize){
	memcpy(hostDst,hostSrc,elements*eleSize);
}

void copyArrayHostToDevice(void* hostSrc, void* devDst, length_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(devDst,hostSrc,elements*eleSize,cudaMemcpyHostToDevice));
}

void copyArrayDeviceToHost(void* devSrc, void* hostDst, length_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(hostDst,devSrc,elements*eleSize,cudaMemcpyDeviceToHost));
}

void copyArrayDeviceToDevice(void* devSrc, void* devDst, length_t elements, int32_t eleSize){
	checkCudaErrors(cudaMemcpy(devDst,devSrc,elements*eleSize,cudaMemcpyDeviceToDevice));
}


