#ifndef _CU_UTILS_INCLUDE_H
#define _CU_UTILS_INCLUDE_H

#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#include "cuStingerDefs.hpp"

bool hasOption(const char* option, int argc, char **argv);
void readGraphDIMACS(char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, int isRmat=0);
void readGraphSNAP  (char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, bool undirected=false);
void readGraphMatrixMarket(char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, bool undirected=false);

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

#ifdef DEBUG
#define _DEBUG(x) x
#else
#define _DEBUG(x)
#endif

void start_clock(cudaEvent_t &start, cudaEvent_t &end);
float end_clock(cudaEvent_t &start, cudaEvent_t &end);



#endif
