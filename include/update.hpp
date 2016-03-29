#pragma once

#include "main.hpp"


class BatchUpdateData{
public:
	BatchUpdateData(length_t batchSize_, bool isHost_){
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
		batchSize=(length_t*) (mem + pos); pos+=sizeof(length_t);	

		*incCount=0;
		*dupCount=0;
		*batchSize=batchSize_;

		if(!isHost){
			dPtr=(BatchUpdateData*) allocDeviceArray(1,sizeof(BatchUpdateData));
			copyArrayHostToDevice(this,dPtr,1, sizeof(BatchUpdateData));
		}
		cout << "The size of the allocated memory is : " << numberBytes << endl;
	}

	~BatchUpdateData(){
		if(isHost){
			freeHostArray(mem);
		}
		else{
			free(dPtr);
			freeDeviceArray(mem);
		}
		cout << "free successfully called" << endl;
	}

	__device__ BatchUpdateData* devicePtr(){return dPtr;}


	__host__ __device__ vertexId_t* getSrc(){return edgeSrc;}	
	__host__ __device__ vertexId_t* getDst(){return edgeDst;}	
	__host__ __device__ vertexId_t* getIndIncomplete(){return indIncomplete;}	
	__host__ __device__ vertexId_t* getIndDuplicate(){return indDuplicate;}	
	__host__ __device__ length_t* getDupPosBatch(){return dupPosBatch;}	
	// Single element values
	__host__ __device__ length_t* getIncCount(){return incCount;}	
	__host__ __device__ length_t* getBatchSize(){return batchSize;}
	__host__ __device__ length_t* getDuplicateCount(){return dupCount;}	

private:
	int8_t* mem;
	int64_t numberBytes;
	bool isHost;


	vertexId_t* edgeSrc;
	vertexId_t* edgeDst;
	vertexId_t* edgeWeight;
	vertexId_t* indIncomplete;
	vertexId_t* indDuplicate;
	length_t* dupPosBatch; 
	// Single element values
	length_t* incCount; 
	length_t* dupCount;
	length_t* batchSize;

	// Used only by device copies of this class
	BatchUpdateData* dPtr;
};



class BatchUpdate{
public:
	BatchUpdate(int32_t batchSize_);
	~BatchUpdate();


	int32_t* getHostSrc(){return h_edgeSrc;}	
	int32_t* getHostDst(){return h_edgeDst;}	
	int32_t* getHostIndIncomplete(){return h_indIncomplete;}	
	int32_t getHostIncCount(){return h_incCount[0];}	
	int32_t getHostBatchSize(){return h_batchSize[0];}
	int32_t* getHostIndDuplicate(){return h_indDuplicate;}	
	int32_t getHostDuplicateCount(){return h_dupCount[0];}	
	int32_t* getHostDupRelPos(){return h_dupRelPos;}	

	
	__device__ int32_t* getDeviceSrc(){return d_edgeSrc;}	
	__device__ int32_t* getDeviceDst(){return d_edgeDst;}	
	__device__ int32_t* getDeviceIndIncomplete(){return d_indIncomplete;}	
	__device__ int32_t* getDeviceIncCount(){return d_incCount;}	
	__device__ int32_t* getDeviceIndDuplicate(){return d_indDuplicate;}	
	__device__ int32_t* getDeviceDuplicateCount(){return d_dupCount;}	
	__device__ int32_t* getDeviceDupRelPos(){return d_dupRelPos;}	


	__device__ int32_t getDeviceBatchSize(){return d_batchSize[0];}

	void resetHostIncCount(){h_incCount[0]=0;}
	void resetDeviceIncCount(){checkCudaErrors(cudaMemset(d_incCount,0,sizeof(int32_t)));}

	void resetHostDuplicateCount(){h_dupCount[0]=0;}
	void resetDeviceDuplicateCount(){checkCudaErrors(cudaMemset(d_dupCount,0,sizeof(int32_t)));}

	BatchUpdate* devicePtr(){return d_batchUpdate;}

	void copyHostToDevice();
	void copyDeviceToHost();

	void copyDeviceToHostDupCount(){copyArrayDeviceToHost(d_dupRelPos,h_dupRelPos,1,sizeof(int32_t));}
	void copyDeviceToHostIncCount(){copyArrayDeviceToHost(d_incCount,h_incCount,1,sizeof(int32_t));}

	void reAllocateMemoryAfterSweep1(cuStinger &custing);

private:

	BatchUpdate* d_batchUpdate;

	int32_t *h_batchSize,*h_edgeSrc,*h_edgeDst, *h_indIncomplete, *h_incCount, *h_indDuplicate, *h_dupRelPos,*h_dupCount;
	int32_t *d_batchSize,*d_edgeSrc,*d_edgeDst, *d_indIncomplete, *d_incCount, *d_indDuplicate, *d_dupRelPos,*d_dupCount;

	BatchUpdateData *hData, *dData;
};



