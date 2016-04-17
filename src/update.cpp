
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


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

BatchUpdateData::BatchUpdateData(length_t batchSize_, bool isHost_){
	isHost = isHost_;
	numberBytes = batchSize_* (5 * sizeof(vertexId_t)+ 1* sizeof(length_t)) + 3*sizeof (length_t);

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

	if(isHost){
		*incCount=0;
		*dupCount=0;
		*batchSize=batchSize_;
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
	cout << "The batch size is :" << batchSize << endl;

	hData = new BatchUpdateData(batchSize,true);
	dData = new BatchUpdateData(batchSize,false);

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

void BatchUpdate::reAllocateMemoryAfterSweep1(cuStinger &custing)
{
	// This function consists of two main phases. 
	// 1) Given the list of edges that could be inserted due to lack of space, this list is reduced 
	// such to a unique set of source vertices (i.e. there might be numerous edges that could not be inserted
	// that have the same source vertex.). 
	// 2) Reallocate edge-data memory on the device and copy the previous version of the edge-data.

	vertexId_t *tempsrc=getHostBUD()->getSrc(),*incomplete = getHostBUD()->getIndIncomplete();
	length_t incCount = *(getHostBUD()->getIncCount());

	// The following hash-map used for finding duplicate source edges.
	unordered_map <vertexId_t, length_t> h_hmap;

	for (length_t i=0; i<incCount; i++){
		vertexId_t temp = tempsrc[incomplete[i]];
		h_hmap[temp]++;
		if(temp==8193)
			cout << incomplete[i] << endl;
	}

	// Contains the list of unique src vertices
	vertexId_t* h_requireUpdates=(vertexId_t*)allocHostArray(*(getHostBUD()->getBatchSize()), sizeof(vertexId_t));
	// For each unique vertex contains how many times that unique source vertex was over the maximal limit
	length_t* h_overLimit=(length_t*)allocHostArray(*(getHostBUD()->getBatchSize()), sizeof(length_t));

	// cout << "The first unique vertex is: "<< tempsrc[incomplete[0]] << endl;
	// Extracting the unique source vertices.
	length_t countUnique=0;
	for (length_t i=0; i<incCount; i++){
		vertexId_t temp = tempsrc[incomplete[i]];
		if(h_hmap[temp]!=0){
			h_requireUpdates[countUnique]=temp;
			h_overLimit[countUnique]=h_hmap[temp];
			countUnique++;
			h_hmap[temp]=0; // Once a vertex is extracted, the hash-map is reset to avoid extracting the source vertex multiple times.
		}
	}

	if(countUnique>0){
		// Allocate memory to store the vertex data before the update.
		// We need this information, especially the pointers to the older edge lists as these need to deallocated 
		// after data is copied to the newly allocated arrays.
		cuStinger::cusVertexData* oldhVD = new cuStinger::cusVertexData();
		vertexId_t nv = custing.getMaxNV();
		oldhVD->hostAllocateMemoryandInitialize(nv,custing.getBytesPerVertex());

		// Copy device VD back to the host. Make an additional host copy for de-allocating memory.
		cuStinger::cusVertexData* cushVD = custing.getHostVertexData();
		copyArrayDeviceToHost(custing.getDeviceVertexDataMemory(),cushVD->mem,nv,custing.getBytesPerVertex());
		copyArrayHostToHost(cushVD->mem,oldhVD->mem,nv,custing.getBytesPerVertex());

		// Allocate the necessary memory on the device for the older VD (and make a copy of that meta-data)
		cuStinger::cusVertexData* olddVD = (cuStinger::cusVertexData*)allocDeviceArray(1, sizeof(cuStinger::cusVertexData));
		uint8_t* olddedmem = (uint8_t*)allocDeviceArray(nv,custing.getBytesPerVertex());
		custing.initVertexDataPointers(olddVD,olddedmem);
		copyArrayHostToDevice(oldhVD->mem,olddedmem,nv,custing.getBytesPerVertex());


		// For each unique vertex allocate new EdgeData
		for (length_t i=0; i<countUnique; i++){
			vertexId_t tempVertex = h_requireUpdates[i];
			length_t newMax = custing.getUpdateAllocater()(cushVD->max[tempVertex] ,h_overLimit[i]);
			cushVD->adj[tempVertex]  	= (cuStinger::cusEdgeData*)allocDeviceArray(1, sizeof(cuStinger::cusEdgeData));
			cushVD->edMem[tempVertex]	= (uint8_t*)allocDeviceArray(newMax, custing.getBytesPerEdge());
			cushVD->max[tempVertex] 	= newMax;
		}

		// Copy the host VD back to STINGER.
		copyArrayHostToDevice(cushVD->mem,custing.dedmem,nv,custing.getBytesPerVertex());

		// Copy unique vertex sources to device
		vertexId_t * d_requireUpdates = (vertexId_t*) allocDeviceArray(countUnique, sizeof(vertexId_t));
		copyArrayHostToDevice(h_requireUpdates,d_requireUpdates,countUnique,sizeof(vertexId_t));


		// Modify the data structure on the device. This includes copying all the data concurrently on the device.
		custing.copyMultipleAdjacencies(olddVD,d_requireUpdates,countUnique);

		// cudaEvent_t ce_start,ce_stop;
		// start_clock(ce_start, ce_stop);

		// De-allocate older ED that is no longer needed.
		for (length_t i=0; i<countUnique; i++){
			vertexId_t tempVertex = h_requireUpdates[i];
			freeDeviceArray(oldhVD->edMem[tempVertex]);
			freeDeviceArray(oldhVD->adj[tempVertex]);
		}
		// cout << "Reallocate time     : " << end_clock(ce_start, ce_stop) << endl;

		// Remove all auxiliary arrays.
		oldhVD->hostFreeMem();
		delete oldhVD;
		freeDeviceArray(olddedmem);
		freeDeviceArray(olddVD);
		freeDeviceArray(d_requireUpdates);
	}
	freeHostArray(h_requireUpdates); 	
	freeHostArray(h_overLimit);
}

	
