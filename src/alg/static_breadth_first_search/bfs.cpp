#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)

typedef void (*cus_kernel_call)(cuStinger& custing, void* func_meta_data);

void bfsMain(cuStinger& custing, void* func_meta_data);
void connectComponentsMain(cuStinger& custing, void* func_meta_data);
void connectComponentsMainLocal(cuStinger& custing, void* func_meta_data);

void oneMoreMain(cuStinger& custing, void* func_meta_data);

int main(const int argc, char *argv[]){
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne,*off;
    vertexId_t *adj;

	bool isDimacs,isSNAP,isRmat=false;
	string filename(argv[1]);
	isDimacs = filename.find(".graph")==std::string::npos?false:true;
	isSNAP   = filename.find(".txt")==std::string::npos?false:true;
	isRmat 	 = filename.find("kron")==std::string::npos?false:true;

	if(isDimacs){
	    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne,isRmat);
	}
	else if(isSNAP){
	    readGraphSNAP(argv[1],&off,&adj,&nv,&ne);
	}
	else{ 
		cout << "Unknown graph type" << endl;
	}

	// cout << "Vertices " << nv << endl;
	// cout << "Edges " << ne << endl;

	cudaEvent_t ce_start,ce_stop;
	cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

	cuStingerInitConfig cuInit;
	cuInit.initState =eInitStateCSR;
	cuInit.maxNV = nv+1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	// CSR data
	cuInit.csrNV 			= nv;
	cuInit.csrNE	   		= ne;
	cuInit.csrOff 			= off;
	cuInit.csrAdj 			= adj;
	cuInit.csrVW 			= NULL;
	cuInit.csrEW			= NULL;

	custing.initializeCuStinger(cuInit);

	cus_kernel_call call_kernel = bfsMain;
	call_kernel(custing,NULL);

	call_kernel = connectComponentsMain;
	call_kernel(custing,NULL);

	call_kernel = connectComponentsMainLocal;
	call_kernel(custing,NULL);

	call_kernel = oneMoreMain;
	call_kernel(custing,NULL);



	custing.freecuStinger();

	free(off);
	free(adj);
    return 0;	
}
