
#include <cuda_runtime.h>


#include <unordered_map>
#include <algorithm>

#include "utils.hpp"
#include "update.hpp"
#include "memoryManager.hpp"
#include "cuStinger.hpp"

using namespace std;

//----------------
//----------------
//----------------
// BatchUpdateData
//----------------
//----------------
//----------------

/// Constructor for BatchUpdateData. 
/// Responsible for allocating the memory on the device.
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


/// Destructor for BatchUpdateData
/// Frees memory on the host and the device.
BatchUpdateData::~BatchUpdateData(){
	if(isHost){
		freeHostArray(mem);
	}
	else{
		freeDeviceArray(dPtr);
		freeDeviceArray(mem);
	}
}


/// Resets the counter responsible for counting vertices that a new adjaecny list.
void BatchUpdateData::resetIncCount(){
	if(isHost){
		*incCount=0;
	}
	else{
		checkCudaErrors(cudaMemset(incCount,0,sizeof(length_t)));
	}
}

/// Resets the duplicate counters responsible for stating how many duplicates exist within a batch.
void BatchUpdateData::resetDuplicateCount(){
	if(isHost){
		*dupCount=0;
	}
	else{
		checkCudaErrors(cudaMemset(dupCount,0,sizeof(length_t)));
	}
}


/// Copies a batch update from one instance to another - both instances are on the host.
void BatchUpdateData::copyHostToHost(BatchUpdateData &hBUA){
	if (isHost && hBUA.isHost){
		copyArrayHostToHost(hBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}

/// Copies a batch update from one instance to another - source batch update is on the host and destiniation is the device.
void BatchUpdateData::copyHostToDevice(BatchUpdateData &hBUA){
	if (!isHost && hBUA.isHost){
		copyArrayHostToDevice(hBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}

/// Copies a batch update from one instance to another - source batch update is on the device and destiniation is the host.
void BatchUpdateData::copyDeviceToHost(BatchUpdateData &dBUA){
	if (isHost && !dBUA.isHost){	
		copyArrayDeviceToHost(dBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}

/// Copies only the duplicate count field from the device to the host 
void BatchUpdateData::copyDeviceToHostDupCount(BatchUpdateData &dBUD){
	if (isHost && !dBUD.isHost){
		copyArrayDeviceToHost(dBUD.dupCount,dupCount,1,sizeof(length_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy device array to host array in ") + string(typeid(*this).name())+ string(". One array is not the right type"));
	}
}

/// Copies only the duplicate count field from the device to the host 
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

	// Creating one batch update one the host and one on the device.
	// Users will use the host version and we will copy the information to the device without the user needing to it explicitily.
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

