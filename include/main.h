// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{   
    if (cudaSuccess != err)
    {   
        std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err) << " from file " << file  << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#ifndef checkLastCudaError
#define checkLastCudaError(str)  __checkLastCudaError (str,__FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkLastCudaError(const char* strError, const char *file, const int line)
{   
	cudaError_t error = cudaGetLastError();
    if (cudaSuccess != error)
    {   
        std::cerr << "Execution error = " << strError << ": " << cudaGetErrorString(error) << " from file " << file  << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif


#pragma once

// #include "update.hpp"
#include "cuStinger.hpp"

typedef int32_t* int32_tPtr;
typedef int32_tPtr* int32_tPtrPtr;


//  void allocGPUMemory(int32_t nv,int32_t ne,int32_t* off, int32_t* adj,
// 	int32_tPtrPtr* d_adj,int32_t** d_utilized,int32_t** d_max);

// void hostMakeGPUStinger(int32_t nv,int32_t ne,int32_t* h_off, int32_t* h_adj,
// 	int32_tPtr* d_adj,int32_t* d_utilized,int32_t* d_max);


void* allocHostArray(int32_t elements,int32_t eleSize);
void* allocDeviceArray(int32_t elements,int32_t eleSize);
void freeHostArray(void* array);
void freeDeviceArray(void* array);

void copyArrayHostToDevice(void* hostSrc, void* devDst, int32_t elements, int32_t eleSize);
void copyArrayDeviceToHost(void* devSrc, void* hostDst, int32_t elements, int32_t eleSize);


// void update(int32_t nv,int32_t ne,
// 	int32_tPtr* d_adj,int32_t* d_utilized,int32_t* d_max,
// 	int32_t numUpdates, int32_t* h_updatesSrc, int32_t* h_updatesDst, 
// 	int32_t* d_updatesSrc, int32_t* d_updatesDst);





