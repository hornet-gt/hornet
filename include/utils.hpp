
#pragma once



void* allocHostArray(int32_t elements,int32_t eleSize);
void* allocDeviceArray(int32_t elements,int32_t eleSize);
void freeHostArray(void* array);
void freeDeviceArray(void* array);

void copyArrayHostToHost    (void* hostSrc,  void* hostDst, int32_t elements, int32_t eleSize);
void copyArrayHostToDevice  (void* hostSrc,  void* devDst,  int32_t elements, int32_t eleSize);
void copyArrayDeviceToHost  (void* devSrc,   void* hostDst, int32_t elements, int32_t eleSize);
void copyArrayDeviceToDevice(void* devSrc,   void* devtDst, int32_t elements, int32_t eleSize);

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
