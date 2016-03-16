
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

	for(int v = 0; v < nv; v++){
        freeDeviceArray(h_adj[v]); 
    }

	freeDeviceArray(d_cuStinger);
	freeDeviceArray(d_adj);
	freeDeviceArray(d_utilized);
	freeDeviceArray(d_max);

	freeHostArray(h_adj);
	freeHostArray(h_utilized);
	freeHostArray(h_max);	
}

void cuStinger::copyHostToDevice(){
	copyArrayHostToDevice(h_utilized,d_utilized,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_max,d_max,nv,sizeof(int32_t));
	copyArrayHostToDevice(h_adj,d_adj,nv,sizeof(int32_t*));
}

void cuStinger::copyDeviceToHost(){
	copyArrayDeviceToHost(d_utilized,h_utilized,nv,sizeof(int32_t));
	copyArrayDeviceToHost(d_max,h_max,nv,sizeof(int32_t));
	copyArrayDeviceToHost(d_adj,h_adj,nv,sizeof(int32_t*));
}


void cuStinger::deviceAllocMemory(int32_t* off, int32_t* adj)
{	
	d_adj = (int32_t**)allocDeviceArray(nv,sizeof(int32_t*));
	d_utilized = (int32_t*)allocDeviceArray(nv,sizeof(int32_t));
	d_max =  (int32_t*)allocDeviceArray(nv,sizeof(int32_t));

	h_adj =  (int32_t**)allocHostArray(nv,sizeof(int32_t*));
	h_utilized =  (int32_t*)allocHostArray(nv,sizeof(int32_t));
	h_max =  (int32_t*)allocHostArray(nv,sizeof(int32_t));

	for(int v=0; v<nv; v++){
		h_utilized[v]=off[v+1]-off[v];
		h_max[v] = elementsPerVertex(h_utilized[v]);
		h_adj[v] =  (int32_t*)allocDeviceArray(h_max[v], sizeof(int32_t));
	}
	copyHostToDevice();
}


void cuStinger::initializeCuStinger(int32_t nv_,int32_t ne_,int32_t* off_, int32_t* adj_){
	nv=nv_;
	ne=ne_;	
	deviceAllocMemory(off_,adj_);

	d_cuStinger=(cuStinger*) allocDeviceArray(1,sizeof(cuStinger));
	copyArrayHostToDevice(this,d_cuStinger,1, sizeof(cuStinger));

	initcuStinger(off_,adj_);
}

__global__ void devMakeGPUStinger(int32_t* d_off, int32_t* d_adj,
	int verticesPerThreadBlock,cuStinger* custing)
{
	int32_t** d_cuadj = custing->d_adj;
	int32_t* d_utilized = custing->d_utilized;

	int32_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		int32_t v=v_init+v_hat;
		if(v>=custing->nv)
			break;
		for(int32_t e=threadIdx.x; e<d_utilized[v]; e+=blockDim.x){
			d_cuadj[v][e]=d_adj[d_off[v]+e];
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



