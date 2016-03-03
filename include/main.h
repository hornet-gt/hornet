#pragma once


typedef int32_t* int32_tPtr;
typedef int32_tPtr* int32_tPtrPtr;


// void allocGPUMemory(int32_t nv,int32_t ne,int32_t* off, int32_t* adj,
// 	thrust::device_vector<int32_t> **d_adjArray,
// 	thrust::device_vector<int32_t> *d_adjSizeUsed,thrust::device_vector<int32_t> d_adjSizeMax);

void allocGPUMemory(int32_t nv,int32_t ne,int32_t* off, int32_t* adj,
	int32_tPtrPtr* d_adjArray,int32_t** d_adjSizeUsed,int32_t** d_adjSizeMax);

void hostMakeGPUStinger(int32_t nv,int32_t ne,int32_t* off, int32_t* adj,
	thrust::device_vector<int32_t> *d_adjArray,thrust::device_vector<int32_t> d_adjSizeUsed);



void* allocHostArray(int32_t elements,int32_t eleSize);
void* allocDeviceArray(int32_t elements,int32_t eleSize);
void freeHostArray(void* array);
void freeDeviceArray(void* array);

void copyArrayHostToDevice(void* hostSrc, void* devDst, int32_t elements, int32_t eleSize);
void copyArrayDeviceToHost(void* devSrc, void* hostDst, int32_t elements, int32_t eleSize);

