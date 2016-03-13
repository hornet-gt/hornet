
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.h"
#include "cuStinger.hpp"

using namespace std;

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

cuStinger::~cuStinger(){
}

void cuStinger::freecuStinger(){

	int32_t** h_adjArray = (int32_t**)allocHostArray(nv, sizeof(int32_t*));
	copyArrayDeviceToHost(d_adjArray,h_adjArray,nv, sizeof(int32_t*));
	for(int v = 0; v < nv; v++){
        freeDeviceArray(h_adjArray[v]); 
    }

	freeDeviceArray(d_cuStinger);
	freeDeviceArray(d_adjArray);
	freeDeviceArray(d_adjSizeUsed);
	freeDeviceArray(d_adjSizeMax);
}

void cuStinger::deviceAllocMemory(int32_t* off, int32_t* adj)
{	
	d_adjArray = (int32_t**)allocDeviceArray(nv,sizeof(int32_t*));

	d_adjSizeUsed = (int32_t*)allocDeviceArray(nv,sizeof(int32_t));
	d_adjSizeMax =  (int32_t*)allocDeviceArray(nv,sizeof(int32_t));

	int32_t** h_arrayPtr =  (int32_t**)allocHostArray(nv,sizeof(int32_t*));
	int32_t* h_sizeArrayUsed =  (int32_t*)allocHostArray(nv,sizeof(int32_t));
	int32_t* h_sizeArrayMax =  (int32_t*)allocHostArray(nv,sizeof(int32_t));

	for(int v=0; v<nv; v++){
		h_sizeArrayUsed[v]=off[v+1]-off[v];
		h_sizeArrayMax[v] = elementsPerVertex(h_sizeArrayUsed[v]);
		h_arrayPtr[v] =  (int32_t*)allocDeviceArray(h_sizeArrayMax[v], sizeof(int32_t));
	}
	copyArrayHostToDevice(h_sizeArrayUsed,d_adjSizeUsed,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_sizeArrayMax,d_adjSizeMax,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_arrayPtr,d_adjArray,nv,sizeof(int32_t*));

	freeHostArray(h_arrayPtr);
	freeHostArray(h_sizeArrayUsed);
	freeHostArray(h_sizeArrayMax);
}


void cuStinger::hostCsrTocuStinger(int32_t nv_,int32_t ne_,int32_t* off_, int32_t* adj_){
	nv=nv_;
	ne=ne_;
	deviceAllocMemory(off_,adj_);

	d_cuStinger=(cuStinger*) allocDeviceArray(1,sizeof(cuStinger));
	copyArrayHostToDevice(this,d_cuStinger,1, sizeof(cuStinger));

	// cuStinger* h_cuStinger=(cuStinger*) allocHostArray(1,sizeof(cuStinger));
	// copyArrayDeviceToHost(d_cuStinger,h_cuStinger,1, sizeof(cuStinger));	
	// printf("nv: %d %d \n",h_cuStinger->nv,nv);	
	// printf("ne: %d %d \n",h_cuStinger->ne,ne);	
	// printf("Adj ptrs: %p %p \n",h_cuStinger->d_adjArray,d_adjArray);	
	// printf("Adj ptrs: %p %p \n",h_cuStinger->d_adjSizeUsed,d_adjSizeUsed);	
	// printf("Adj ptrs: %p %p \n",h_cuStinger->d_adjSizeMax,d_adjSizeMax);	
	// freeDeviceArray(h_cuStinger);

	initcuStinger(off_,adj_);
	cout << "after copying" << endl;
}

__global__ void devMakeGPUStinger(int32_t* d_off, int32_t* d_adj,
	int verticesPerThreadBlock,cuStinger* custing)
{
	int32_t** d_adjArray = custing->d_adjArray;
	int32_t* d_adjSizeUsed = custing->d_adjSizeUsed;

	int32_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		int32_t v=v_init+v_hat;
		if(v>=custing->nv)
			break;
		for(int32_t e=threadIdx.x; e<d_adjSizeUsed[v]; e+=blockDim.x){
			d_adjArray[v][e]=d_adj[d_off[v]+e];
		}
	}
}


void cuStinger::initcuStinger(int32_t* h_off, int32_t* h_adj){
	int32_t* d_off = (int32_t*)allocDeviceArray(nv+1,sizeof(int32_t));
	int32_t* d_adj = (int32_t*)allocDeviceArray(ne,sizeof(int32_t));
	copyArrayHostToDevice(h_off,d_off,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_adj,d_adj,ne,sizeof(int32_t));

	dim3 numBlocks(1, 1);
	int32_t threads=64;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)nv/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	

	int32_t verticesPerThreadBlock = ceil(float(nv)/float(numBlocks.x-1));

	devMakeGPUStinger<<<numBlocks,threadsPerBlock>>>(d_off,d_adj,verticesPerThreadBlock, d_cuStinger);

	freeDeviceArray(d_adj);	
	freeDeviceArray(d_off);
}



