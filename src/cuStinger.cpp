

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>

#include "main.hpp"

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

int32_t exactInitAllocater(int32_t elements){
	return elements;
}

int32_t stingyInitAllocater(int32_t elements){
	return elements+1;
}

int32_t defaultUpdateAllocater(int32_t elements, int32_t overLimit){
	int32_t eleCount = elements+overLimit;
	if(eleCount==0)
		eleCount=1;
	else if(eleCount < 5)
		eleCount*=2;
	else
		eleCount*=1.5;
	return eleCount;
}

int32_t exactUpdateAllocater(int32_t elements, int32_t overLimit){
	return elements+overLimit;
}

int32_t stingyUpdateAllocater(int32_t elements, int32_t overLimit){
	return elements+overLimit+1;
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
	freeDeviceArray(d_vweight);
	freeDeviceArray(d_vtype);

	freeHostArray(h_adj);
	freeHostArray(h_utilized);
	freeHostArray(h_max);
	freeHostArray(h_vweight);
	freeHostArray(h_vtype);

}

void cuStinger::copyHostToDevice(){
	copyArrayHostToDevice(h_utilized,d_utilized,nv,sizeof(length_t));
	copyArrayHostToDevice(h_max,d_max,nv,sizeof(length_t));
	copyArrayHostToDevice(h_adj,d_adj,nv,sizeof(int32_t*));
	copyArrayHostToDevice(h_vweight,d_vweight,nv,sizeof(vweight_t));
	copyArrayHostToDevice(h_vtype,d_vtype,nv,sizeof(vtype_t));
}

void cuStinger::copyDeviceToHost(){
	copyArrayDeviceToHost(d_utilized,h_utilized,nv,sizeof(length_t));
	copyArrayDeviceToHost(d_max,h_max,nv,sizeof(length_t));
	copyArrayDeviceToHost(d_adj,h_adj,nv,sizeof(int32_t*));
	copyArrayDeviceToHost(d_vweight,h_vweight,nv,sizeof(vweight_t));
	copyArrayDeviceToHost(d_vtype,h_vtype,nv,sizeof(vtype_t));
}

void cuStinger::deviceAllocMemory(int32_t* off, int32_t* adj)
{	
	d_adj = (int32_t**)allocDeviceArray(nv,sizeof(int32_t*));
	d_utilized = (length_t*)allocDeviceArray(nv,sizeof(length_t));
	d_max =  (length_t*)allocDeviceArray(nv,sizeof(length_t));
	d_vweight = (vweight_t*)allocDeviceArray(nv,sizeof(vweight_t));
	d_vtype = (vtype_t*)allocDeviceArray(nv,sizeof(vtype_t));

	h_adj =  (int32_t**)allocHostArray(nv,sizeof(int32_t*));
	h_utilized =  (length_t*)allocHostArray(nv,sizeof(length_t));
	h_max =  (length_t*)allocHostArray(nv,sizeof(length_t));
	h_vweight = (vweight_t*)allocHostArray(nv,sizeof(vweight_t));
	h_vtype = (vtype_t*)allocHostArray(nv,sizeof(vtype_t));

	for(int v=0; v<nv; v++){
		h_utilized[v]=off[v+1]-off[v];
		h_max[v] = initVertexAllocator(h_utilized[v]);
		h_adj[v] =  (int32_t*)allocDeviceArray(h_max[v], sizeof(int32_t));

	}
	copyHostToDevice();
}


void cuStinger::initializeCuStinger(int32_t nv_,int32_t ne_,int32_t* off_, int32_t* adj_){
	nv=nv_;
	deviceAllocMemory(off_,adj_);

	d_cuStinger=(cuStinger*) allocDeviceArray(1,sizeof(cuStinger));
	copyArrayHostToDevice(this,d_cuStinger,1, sizeof(cuStinger));

	internalInitcuStinger(off_,adj_,ne_);
}


	// void initializeCuStinger(cuStingerConfig);



int32_t cuStinger::getNumberEdgesAllocated(){
	return sumDeviceArray(d_max);
}

int32_t cuStinger::getNumberEdgesUsed(){
	return sumDeviceArray(d_utilized);
}


