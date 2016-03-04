 

//#include <iostream>
//#include <numeric>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.h"

using namespace std;

void readGraphDIMACS(char* filePath, int32_t** prmoff, int32_t** prmind, int32_t* prmnv, int32_t* prmne);

int main(const int argc, char *argv[])
{
    cudaSetDevice(0);
	// cudaDeviceProp prop;
	// cudaGetDeviceProperties(&prop, 0);
 	// printf("  Device name: %s\n", prop.name);

    int32_t nv, ne,*off,*adj;

    cout << argv[1] << endl;
    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne);

	cout << "Vertices " << nv << endl;
	cout << "Edges " << ne << endl;

	int32_t *d_adjSizeUsed,*d_adjSizeMax,**d_adjArray;

	allocGPUMemory(nv, ne, off, adj, &d_adjArray, &d_adjSizeUsed, &d_adjSizeMax);

	int32_t** h_adjArray = (int32_t**)allocHostArray(nv, sizeof(int32_t*));

	copyArrayDeviceToHost(d_adjArray,h_adjArray,nv, sizeof(int32_t*));

	cout << "baabaa" << endl;

	for(int v = 0; v < nv; v++){
        freeDeviceArray(h_adjArray[v]); 
    }

	freeDeviceArray(d_adjArray);
	freeDeviceArray(d_adjSizeUsed);
	freeDeviceArray(d_adjSizeMax);

    return 0;	cout << "baabaa" << endl;

}       

