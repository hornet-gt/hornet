

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>

#include "main.h"
#include "cuStinger.hpp"

using namespace std;

int32_t defaultInitAllocater(int32_t elements){
	int32_t eleCount = elements;
	if(eleCount==0)
		eleCount=1;
	else if(eleCount < 5)
		eleCount*=2;
	else
		eleCount*=1.5;
	return eleCount;
}

cuStinger::cuStinger(initAllocator iAllocator,updateAllocator uAllocator){
	initVertexAllocator = iAllocator;
	updateVertexAllocator = uAllocator;
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
		h_max[v] = initVertexAllocator(h_utilized[v]);
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



int32_t cuStinger::getNumberEdgesAllocated(){
	return sumDeviceArray(d_max);
}

int32_t cuStinger::getNumberEdgesUsed(){
	return sumDeviceArray(d_utilized);
}


