
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
	// numberBytes = sizeof(BatchUpdateData);
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
		// copyArrayHostToDevice(this,dPtr,1, sizeof(BatchUpdateData));

	batchSize=(length_t*) (mem + pos); pos+=sizeof(length_t);	
	// cout << "Pos is  " << pos << endl; 

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
	int32_t sum=0; 
	vertexId_t *tempsrc=getHostBUD()->getSrc(),*tempdst=getHostBUD()->getDst(),*incomplete = getHostBUD()->getIndIncomplete();
	length_t incCount = *(getHostBUD()->getIncCount());

	unordered_map <vertexId_t, length_t> h_hmap;

	vertexId_t* h_requireUpdates=(vertexId_t*)allocHostArray(*(getHostBUD()->getBatchSize()), sizeof(vertexId_t));
	length_t* h_overLimit=(length_t*)allocHostArray(*(getHostBUD()->getBatchSize()), sizeof(length_t));

	for (length_t i=0; i<incCount; i++){
		vertexId_t temp = tempsrc[incomplete[i]];
		h_hmap[temp]++;
	}

	length_t countUnique=0;
	for (length_t i=0; i<incCount; i++){
		vertexId_t temp = tempsrc[incomplete[i]];
		if(h_hmap[temp]!=0){
			h_requireUpdates[countUnique]=temp;
			h_overLimit[countUnique]=h_hmap[temp];
			countUnique++;
			h_hmap[temp]=0;
		}
	}

	cudaEvent_t ce_start,ce_stop;


	if(countUnique>0){

		cuStinger::cusVertexData* oldhVD = new cuStinger::cusVertexData();
		vertexId_t nv = custing.getMaxNV();
		oldhVD->hostAllocateMemoryandInitialize(nv,custing.getBytesPerVertex());
		cuStinger::cusVertexData* cushVD = custing.getHostVertexData();

		start_clock(ce_start, ce_stop);
			copyArrayDeviceToHost(custing.getDeviceVertexDataMemory(),oldhVD->mem,nv,custing.getBytesPerVertex());
			copyArrayHostToHost(oldhVD->mem,cushVD->mem,nv,custing.getBytesPerVertex());
		cout << "Copy time from device to host of util arrays : " << end_clock(ce_start, ce_stop) << endl;

		cuStinger::cusVertexData* olddVD = (cuStinger::cusVertexData*)allocDeviceArray(1, sizeof(cuStinger::cusVertexData));
		uint8_t* olddedmem = (uint8_t*)allocDeviceArray(nv,custing.getBytesPerVertex());
		custing.initVertexDataPointers(olddVD,olddedmem);
		// copyArrayHostToDevice(oldhVD->mem,olddedmem,nv,custing.getBytesPerVertex());
		copyArrayDeviceToDevice(custing.dedmem,olddedmem,nv,custing.getBytesPerVertex());



		for (length_t i=0; i<countUnique; i++){
			vertexId_t tempVertex = h_requireUpdates[i];
			length_t newMax = custing.getUpdateAllocater()(cushVD->max[tempVertex] ,h_overLimit[i]);
			cushVD->adj[tempVertex]  	= (cuStinger::cusEdgeData*)allocDeviceArray(1, sizeof(cuStinger::cusEdgeData));
			cushVD->edMem[tempVertex]	= (uint8_t*)allocDeviceArray(newMax, custing.getBytesPerEdge());

			// cushVD->used[tempVertex] 	= cushVD->max[tempVertex];
			cushVD->max[tempVertex] 	= newMax;
			if(i==0){
				cout << "After  renewing : " << tempVertex << "   " << cushVD->used[tempVertex] << "   " << cushVD->max[tempVertex] << endl;
			}

		}

		copyArrayHostToDevice(cushVD->mem,custing.dedmem,nv,custing.getBytesPerVertex());


		vertexId_t * d_requireUpdates = (vertexId_t*) allocDeviceArray(countUnique, sizeof(vertexId_t));
		copyArrayHostToDevice(h_requireUpdates,d_requireUpdates,countUnique,sizeof(vertexId_t));


		custing.copyMultipleAdjacencies(olddVD,d_requireUpdates,countUnique);

		for (length_t i=0; i<countUnique; i++){
			vertexId_t tempVertex = h_requireUpdates[i];
			freeDeviceArray(oldhVD->edMem[tempVertex]);
			freeDeviceArray(oldhVD->adj[tempVertex]);
		}
		oldhVD->hostFreeMem();
		delete oldhVD;

		freeDeviceArray(olddedmem);
		freeDeviceArray(olddVD);

		freeDeviceArray(d_requireUpdates);

	}

	freeHostArray(h_requireUpdates);
	freeHostArray(h_overLimit);
}

	