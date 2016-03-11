#pragma once

#include "main.h"

class BatchUpdate{
public:
	BatchUpdate(int32_t batchSize_);
	~BatchUpdate();


	int32_t* getHostSrcArray(){return h_edgeSrc;}	
	int32_t* getHostDstArray(){return h_edgeDst;}	
	int32_t* getHostIndInCompleteArray(){return h_indIncomplete;}	
	int32_t getHostIndCount(){return h_indCount[0];}	
	int32_t getHostBatchSize(){return h_batchSize[0];}
	
	__device__ int32_t* getDeviceSrcArray(){return d_edgeSrc;}	
	__device__ int32_t* getDeviceDstArray(){return d_edgeDst;}	
	__device__ int32_t* getDeviceIndInCompleteArray(){return d_indIncomplete;}	
	__device__ int32_t* getDeviceIndCount(){return d_indCount;}	
	__device__ int32_t getDeviceBatchSize(){return d_batchSize[0];}

	void resetHostIndCount(){h_indCount[0]=0;}
	void resetDeviceIndCount(){
		checkCudaErrors(cudaMemset(d_indCount,0,sizeof(int32_t)));
	}

	BatchUpdate* devicePtr(){return d_batchUpdate;}

	void copyHostToDevice();
	void copyDeviceToHost();
private:

	BatchUpdate* d_batchUpdate;

	int32_t* h_batchSize;
	int32_t *h_edgeSrc,*h_edgeDst, *h_indIncomplete, *h_indCount;
	int32_t *d_edgeSrc,*d_edgeDst, *d_indIncomplete, *d_indCount;
	int32_t *d_batchSize;

};

