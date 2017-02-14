

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>

#include "utils.hpp"
#include "update.hpp"
#include "memoryManager.hpp"
#include "cuStinger.hpp"

using namespace std;


cuStinger::cuStinger(initAllocator iAllocator,updateAllocator uAllocator){
	initVertexAllocator = iAllocator;
	updateVertexAllocator = uAllocator;
}

cuStinger::~cuStinger(){
}

void cuStinger::freecuStinger(){

	// for(vertexId_t v=0; v<nv; v++){
	// 	// freeDeviceArray(hVD->edMem[v]);
	// 	// freeDeviceArray(hVD->adj[v]);
	// }

	freeDeviceArray(d_cuStinger);
	hVD->hostFreeMem();
	delete hVD;
	freeDeviceArray(dedmem);
	freeDeviceArray(dVD);

	freeHostArray(hMemManEB);
	delete cusMemMan;
}

// void cuStinger::copyHostToDevice(){
// 	cout << "ODED " << __PRETTY_FUNCTION__ << " is not implemented" << endl;
// }

// void cuStinger::copyDeviceToHost(){
// 	cout << "ODED " << __PRETTY_FUNCTION__ << " is not implemented" << endl;
// }

void cuStinger::deviceAllocMemory(length_t* off, vertexId_t* adj)
{	
	cout << "ODED " << __PRETTY_FUNCTION__ << " is not implemented" << endl;
}

void cuStinger::internalEmptyTocuStinger(int NV){
	cout << "ODED " << __PRETTY_FUNCTION__ << " is not implemented" << endl;
}


void cuStinger::initializeCuStinger(length_t nv_,length_t ne_,length_t* off_, vertexId_t* adj_){

	bytesPerEdge = sizeof (vertexId_t);
	if(isSemantic){
		bytesPerEdge += sizeof(eweight_t) + sizeof(etype_t) + 2*sizeof(timestamp_t);
	}
	else if (useEWeight){
		bytesPerEdge += sizeof(eweight_t);
	}
	// cout << "Size of bytesPerEdge = " << bytesPerEdge << endl;

	bytesPerVertex = sizeof(cusEdgeData*) + sizeof (uint8_t*)+ sizeof (length_t) + sizeof (length_t);
	if(isSemantic){
		bytesPerVertex += sizeof(vweight_t) + sizeof(vtype_t);
	}
	else if (useVWeight){
		bytesPerVertex += sizeof(vweight_t);
	}
	// cout << "Size of bytesPerVertex = " << bytesPerVertex << endl;

	nv=nv_;

	hVD = new cusVertexData();
	hVD->hostAllocateMemoryandInitialize(nv,bytesPerVertex);

	dVD = (cusVertexData*)allocDeviceArray(1, sizeof(cusVertexData));
	dedmem = (uint8_t*)allocDeviceArray(nv,bytesPerVertex);


	for(vertexId_t v=0; v<nv; v++){
		hVD->used[v]		= off_[v+1]-off_[v];
		hVD->max[v] 		= initVertexAllocator(hVD->used[v]);
		// hVD->adj[v] 		= (cusEdgeData*)allocDeviceArray(1, sizeof(cusEdgeData));
		// checkLastCudaError("Error initializing data - pointer data");
		// hVD->edMem[v]	 	= (uint8_t*)allocDeviceArray(hVD->max[v], bytesPerEdge);
		// checkLastCudaError("Error initializing data - adjacency list");

		int memSizeOffsetAdj = sizeof(cusEdgeData)/cudaMemManAlignment + cudaMemManAlignment*(sizeof(cusEdgeData)%cudaMemManAlignment>0);
		int memSizeOffsetedMem = cudaMemManAlignment * (int)ceil ((double) (hVD->max[v]* bytesPerEdge) /(double)cudaMemManAlignment);

		memAllocInfo mai = cusMemMan->allocateMemoryBlock(memSizeOffsetAdj+ memSizeOffsetedMem,v);
		hVD->adj[v] = (cusEdgeData*)mai.ptr;
		hVD->edMem[v] = (uint8_t*)(mai.ptr+memSizeOffsetAdj);
		hMemManEB[v] = (edgeBlock*)mai.eb;
		// (edgeBlock*)mai.eb;
		// cout << memSizeOffsetAdj << ", " << memSizeOffsetedMem << ", " << hVD->max[v]* bytesPerEdge << endl ;
		// if (v<10)
		// 	printf("%p %p \n", (cusEdgeData*)mai.ptr , (cusEdgeData*)(mai.ptr+ memSizeOffsetedMem)) ;

	}

	d_cuStinger=(cuStinger*)allocDeviceArray(1,sizeof(cuStinger));
	copyArrayHostToDevice(this,d_cuStinger,1,sizeof(cuStinger));

	initVertexDataPointers(dVD,dedmem);
	// fflush(stdout);
 
	copyArrayHostToDevice(hVD->mem,dedmem,nv,bytesPerVertex);

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
		hMemManEB = (edgeBlock**)allocHostArray(nv, sizeof(edgeBlock*));
		cusMemMan = new memoryManager(cuCS.defaultBlockSize);

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

length_t cuStinger::getNumberEdgesUsed(){
	int32_t pos=(sizeof(cusEdgeData*)+sizeof(uint8_t*))*nv;
	return sumDeviceArray( (length_t*)(dedmem+ pos) ,nv);
}

length_t cuStinger::getNumberEdgesAllocated(){
	int32_t pos=(sizeof(cusEdgeData*)+sizeof(uint8_t*)+sizeof(length_t))*nv;
	return sumDeviceArray( (length_t*)(dedmem+ pos) ,nv);
}



///---------------------------------
///---------------------------------
/// Allocater functionality
///---------------------------------
///---------------------------------



length_t defaultInitAllocater(length_t elements){
	length_t eleCount = elements;
	if(eleCount==0)
		eleCount=10;
	else if(eleCount < 5)
		eleCount=10;
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
		eleCount=3;
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
