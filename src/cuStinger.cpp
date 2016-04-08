

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>

#include "main.hpp"

using namespace std;


cuStinger::cuStinger(initAllocator iAllocator,updateAllocator uAllocator){
	initVertexAllocator = iAllocator;
	updateVertexAllocator = uAllocator;
}

cuStinger::~cuStinger(){
}

void cuStinger::freecuStinger(){
	// for(int v = 0; v < nv; v++){
 //        freeDeviceArray(h_adj[v]); 
 //    }

	for(vertexId_t v=0; v<nv; v++){
		freeDeviceArray(hVD->edMem[v]);
		freeDeviceArray(hVD->adj[v]);
	}

	freeDeviceArray(d_cuStinger);

	// freeDeviceArray(d_adj);
	// freeDeviceArray(d_utilized);
	// freeDeviceArray(d_max);
	// freeDeviceArray(d_vweight);
	// freeDeviceArray(d_vtype);

	// freeHostArray(h_adj);
	// freeHostArray(h_utilized);
	// freeHostArray(h_max);
	// freeHostArray(h_vweight);
	// freeHostArray(h_vtype);

	freeHostArray(hVD->mem);
	// freeDeviceArray(dVD->mem);

	delete hVD;
	freeDeviceArray(dVD);
}

void cuStinger::copyHostToDevice(){
	// copyArrayHostToDevice(h_utilized,d_utilized,nv,sizeof(length_t));
	// copyArrayHostToDevice(h_max,d_max,nv,sizeof(length_t));
	// copyArrayHostToDevice(h_adj,d_adj,nv,sizeof(vertexId_t*));
	// copyArrayHostToDevice(h_vweight,d_vweight,nv,sizeof(vweight_t));
	// copyArrayHostToDevice(h_vtype,d_vtype,nv,sizeof(vtype_t));
}

void cuStinger::copyDeviceToHost(){
	// copyArrayDeviceToHost(d_utilized,h_utilized,nv,sizeof(length_t));
	// copyArrayDeviceToHost(d_max,h_max,nv,sizeof(length_t));
	// copyArrayDeviceToHost(d_adj,h_adj,nv,sizeof(vertexId_t*));
	// copyArrayDeviceToHost(d_vweight,h_vweight,nv,sizeof(vweight_t));
	// copyArrayDeviceToHost(d_vtype,h_vtype,nv,sizeof(vtype_t));
}

void cuStinger::deviceAllocMemory(length_t* off, vertexId_t* adj)
{	
	// d_adj 		= (vertexId_t**)allocDeviceArray(nv,sizeof(vertexId_t*));
	// d_utilized 	= (length_t*)allocDeviceArray(nv,sizeof(length_t));
	// d_max 		= (length_t*)allocDeviceArray(nv,sizeof(length_t));
	// d_vweight 	= (vweight_t*)allocDeviceArray(nv,sizeof(vweight_t));
	// d_vtype 	= (vtype_t*)allocDeviceArray(nv,sizeof(vtype_t));

	// h_adj 		= (vertexId_t**)allocHostArray(nv,sizeof(vertexId_t*));
	// h_utilized 	= (length_t*)allocHostArray(nv,sizeof(length_t));
	// h_max		= (length_t*)allocHostArray(nv,sizeof(length_t));
	// h_vweight 	= (vweight_t*)allocHostArray(nv,sizeof(vweight_t));
	// h_vtype 	= (vtype_t*)allocHostArray(nv,sizeof(vtype_t));

	// for(vertexId_t v=0; v<nv; v++){
	// 	h_utilized[v]=off[v+1]-off[v];
	// 	h_max[v] = initVertexAllocator(h_utilized[v]);
	// 	h_adj[v] =  (vertexId_t*)allocDeviceArray(h_max[v], bytesPerEdge);
	// }
	// copyHostToDevice();
}

// void cuStinger::internalEmptyTocuStinger(int NV){

// }


void cuStinger::initializeCuStinger(length_t nv_,length_t ne_,length_t* off_, int32_t* adj_){

	bytesPerEdge = sizeof (vertexId_t);
	if(isSemantic){
		bytesPerEdge += sizeof(eweight_t) + sizeof(etype_t) + 2*sizeof(timestamp_t);
	}
	else if (useEWeight){
		bytesPerEdge += sizeof(eweight_t);
	}
	cout << "Size of bytesPerEdge = " << bytesPerEdge << endl;

	bytesPerVertex = sizeof(cusEdgeData*) + sizeof (uint8_t*)+ sizeof (length_t) + sizeof (length_t);
	if(isSemantic){
		bytesPerVertex += sizeof(vweight_t) + sizeof(vtype_t);
	}
	else if (useVWeight){
		bytesPerVertex += sizeof(vweight_t);
	}
	cout << "Size of bytesPerVertex = " << bytesPerVertex << endl;

	nv=nv_;

	// uint8_t** h_memPtr = (uint8_t**)allocHostArray(nv, sizeof(uint8_t*));

	hVD = new cusVertexData();
	hVD->mem = (uint8_t*)allocHostArray(nv,bytesPerVertex);
	int32_t pos=0;
	hVD->adj 		= (cusEdgeData**)(hVD->mem + pos); 	pos+=sizeof(cusEdgeData*)*nv;
	hVD->edMem 		= (uint8_t**)(hVD->mem + pos); 		pos+=sizeof(uint8_t*)*nv;
	hVD->used 		= (length_t*)(hVD->mem + pos); 		pos+=sizeof(length_t)*nv;
	hVD->max        = (length_t*)(hVD->mem + pos); 		pos+=sizeof(length_t)*nv;
	hVD->vw         = (vweight_t*)(hVD->mem + pos); 	pos+=sizeof(vweight_t)*nv;
	hVD->vt         = (vtype_t*)(hVD->mem + pos); 		pos+=sizeof(vtype_t)*nv;

	// dVD = new cusVertexData();
	dVD = (cusVertexData*)allocDeviceArray(1, sizeof(cusVertexData));

	uint8_t* temp = (uint8_t*)allocDeviceArray(nv,bytesPerVertex);
	// dVD->adj 		= (cusEdgeData**)(dVD->mem + pos); 	pos+=sizeof(cusEdgeData*)*nv;
	// dVD->edMem 		= (uint8_t**)(dVD->mem + pos); 		pos+=sizeof(uint8_t*)*nv;
	// dVD->used 		= (length_t*)(dVD->mem + pos); 		pos+=sizeof(length_t)*nv;
	// dVD->max        = (length_t*)(dVD->mem + pos); 		pos+=sizeof(length_t)*nv;
	// dVD->vw         = (vweight_t*)(dVD->mem + pos); 	pos+=sizeof(vweight_t)*nv;
	// dVD->vt         = (vtype_t*)(dVD->mem + pos); 		pos+=sizeof(vtype_t)*nv;

	for(vertexId_t v=0; v<nv; v++){
		hVD->used[v]		= off_[v+1]-off_[v];
		hVD->max[v] 		= initVertexAllocator(hVD->used[v]);
		hVD->adj[v] 		= (cusEdgeData*)allocDeviceArray(1, sizeof(cusEdgeData));
		hVD->edMem[v]	 	= (uint8_t*)allocDeviceArray(hVD->max[v], bytesPerEdge);
	}

	// deviceAllocMemory(off_,adj_);
	// printf("Vertex: From the device : %p \n",dVD); fflush(stdout);
	// printf("Vertex: From the device : %p \n",temp); fflush(stdout);
	d_cuStinger=(cuStinger*)allocDeviceArray(1,sizeof(cuStinger));
	copyArrayHostToDevice(this,d_cuStinger,1,sizeof(cuStinger));

	initVertexDataPointers(temp);
	fflush(stdout);

	// cout << "Number of bytes copied : " << nv*bytesPerVertex << endl; 
	copyArrayHostToDevice(hVD->mem,temp,nv,bytesPerVertex);

	// printf("From the host : %p \n",dVD);

	initEdgeDataPointers();

	internalCSRTocuStinger(off_,adj_,ne_);
}


void cuStinger::initializeCuStinger(cuStingerInitConfig &cuCS){
	isSemantic=cuCS.isSemantic;
	useVWeight=cuCS.useVWeight;
	useEWeight=cuCS.useEWeight;

	nv = cuCS.maxNV;

	if(cuCS.initState==eInitStateEmpty){

	}
	else if(cuCS.initState==eInitStateCSR){
		if (cuCS.maxNV<cuCS.csrNV){
			nv=cuCS.csrNV;
			CUSTINGER_WARNING("In the initialization of cuStinger with a CSR graph a maximal NV smaller than the CSR's NV was given")
		}
		initializeCuStinger(cuCS.csrNV, cuCS.csrNE, cuCS.csrOff, cuCS.csrAdj);


	}
	else if(cuCS.initState==eInitStateEdgeList){
		CUSTINGER_ERROR("No support for edge list initialization just yet");
		exit(0);
	}
	else{
		CUSTINGER_ERROR("An illegal state was given to the cuStinger initialization function");
		exit(0);
	}
}

length_t cuStinger::getNumberEdgesAllocated(){
	return sumDeviceArray(dVD->max,nv);
}

length_t cuStinger::getNumberEdgesUsed(){
	return sumDeviceArray(dVD->used,nv);
}








length_t defaultInitAllocater(length_t elements){
	length_t eleCount = elements;
	if(eleCount==0)
		eleCount=1;
	else if(eleCount < 5)
		eleCount*=2;
	else
		eleCount*=1.5;
	return eleCount;
}

length_t exactInitAllocater(length_t elements){
	return elements;
}

length_t stingyInitAllocater(length_t elements){
	return elements+1;
}

length_t defaultUpdateAllocater(length_t elements, length_t overLimit){
	length_t eleCount = elements+overLimit;
	if(eleCount==0)
		eleCount=1;
	else if(eleCount < 5)
		eleCount*=2;
	else
		eleCount*=1.5;
	return eleCount;
}

length_t exactUpdateAllocater(length_t elements, length_t overLimit){
	return elements+overLimit;
}

length_t stingyUpdateAllocater(length_t elements, length_t overLimit){
	return elements+overLimit+1;
}
