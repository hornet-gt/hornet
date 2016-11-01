
#include <cuda_runtime.h>


#include <unordered_map>
#include <algorithm>

#include "main.hpp"


using namespace std;

//----------------
//----------------
//----------------
// BatchUpdateData
//----------------
//----------------
//----------------

BatchUpdateData::BatchUpdateData(length_t batchSize_, bool isHost_, length_t nv_){
	isHost = isHost_;
	numberBytes = batchSize_* (5 * sizeof(vertexId_t)+ 1* sizeof(length_t)) + 
                  4*sizeof (length_t) +
                  nv_ * (2 * sizeof(length_t)) + 1;

	if(isHost){
		mem = (int8_t*)allocHostArray(numberBytes,sizeof(int8_t));
	}
	else{
		mem = (int8_t*)allocDeviceArray(numberBytes,sizeof(int8_t));
	}

	length_t pos=0;
	edgeSrc=(vertexId_t*) (mem + pos); pos+=batchSize_*sizeof(vertexId_t);
	edgeDst=(vertexId_t*) (mem + pos); pos+=batchSize_*sizeof(vertexId_t);
	edgeWeight=(vertexId_t*) (mem + pos); pos+=batchSize_*sizeof(vertexId_t);
	indIncomplete=(vertexId_t*) (mem + pos); pos+=batchSize_*sizeof(vertexId_t);
	indDuplicate=(vertexId_t*) (mem + pos); pos+=batchSize_*sizeof(vertexId_t);
	dupPosBatch=(length_t*) (mem + pos); pos+=batchSize_*sizeof(length_t);

	incCount=(length_t*) (mem + pos); pos+=sizeof(length_t);
	dupCount=(length_t*) (mem + pos); pos+=sizeof(length_t);
	batchSize=(length_t*) (mem + pos); pos+=sizeof(length_t);	
	nv=(length_t*) (mem + pos); pos+=sizeof(length_t);	
	
	offsets=(length_t*) (mem + pos); pos+=(nv_+1)*sizeof(length_t);	
	vNumDuplicates=(length_t*) (mem + pos); pos+=nv_*sizeof(length_t);	

	if(isHost){
		*incCount=0;
		*dupCount=0;
		*batchSize=batchSize_;
		*nv = nv_;
	}
	if(!isHost){
		dPtr=(BatchUpdateData*) allocDeviceArray(1,sizeof(BatchUpdateData));
		copyArrayHostToDevice(this,dPtr,1, sizeof(BatchUpdateData));
	}
}

BatchUpdateData::~BatchUpdateData(){
	if(isHost){
		freeHostArray(mem);
	}
	else{
		freeDeviceArray(dPtr);
		freeDeviceArray(mem);
	}
}


void BatchUpdateData::resetIncCount(){
	if(isHost){
		*incCount=0;
	}
	else{
		checkCudaErrors(cudaMemset(incCount,0,sizeof(length_t)));
	}
}

void BatchUpdateData::resetDuplicateCount(){
	if(isHost){
		*dupCount=0;
	}
	else{
		checkCudaErrors(cudaMemset(dupCount,0,sizeof(length_t)));
	}
}


void BatchUpdateData::copyHostToHost(BatchUpdateData &hBUA){
	if (isHost && hBUA.isHost){
		copyArrayHostToHost(hBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}

void BatchUpdateData::copyHostToDevice(BatchUpdateData &hBUA){
	if (!isHost && hBUA.isHost){
		copyArrayHostToDevice(hBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}
void BatchUpdateData::copyDeviceToHost(BatchUpdateData &dBUA){
	if (isHost && !dBUA.isHost){	
		copyArrayDeviceToHost(dBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}

void BatchUpdateData::copyDeviceToHostDupCount(BatchUpdateData &dBUD){
	if (isHost && !dBUD.isHost){
		copyArrayDeviceToHost(dBUD.dupCount,dupCount,1,sizeof(length_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy device array to host array in ") + string(typeid(*this).name())+ string(". One array is not the right type"));
	}
}

void BatchUpdateData::copyDeviceToHostIncCount(BatchUpdateData &dBUD)
{
	if (isHost && !dBUD.isHost){
		copyArrayDeviceToHost(dBUD.incCount,incCount,1,sizeof(length_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy device array to host array in ") + string(typeid(*this).name())+ string(". One array is not the right type"));
	}
}


//------------
//------------
//------------
// BatchUpdate
//------------
//------------
//------------


BatchUpdate::BatchUpdate(BatchUpdateData &h_bua){

	if(!h_bua.getisHost()){
		CUSTINGER_ERROR(string(typeid(*this).name()) + string(" expects to receive an update list that is host size"));
	}

	length_t batchSize = *(h_bua.getBatchSize());
	length_t nv = *(h_bua.getNumVertices());

	hData = new BatchUpdateData(batchSize,true, nv);
	dData = new BatchUpdateData(batchSize,false, nv);

	hData->copyHostToHost(h_bua);
	dData->copyHostToDevice(h_bua);

	dPtr=(BatchUpdate*) allocDeviceArray(1,sizeof(BatchUpdate));
	copyArrayHostToDevice(this,dPtr,1, sizeof(BatchUpdate));
}

BatchUpdate::~BatchUpdate(){
	freeDeviceArray(dPtr);

	delete hData;
	delete dData;
}

