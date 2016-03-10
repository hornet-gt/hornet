#pragma once

#include "main.h"

class BatchUpdate{
public:
	BatchUpdate(int32_t batchSize_);
	~BatchUpdate();

	int32_t getBatchSize(){return batchSize;}

	int32_t* getHostSrcArray(){return h_edgeSrc;}	
	int32_t* getHostDstArray(){return h_edgeDst;}	
	int32_t* getHostIndInCompleteArray(){return h_indIncomplete;}	
	int32_t* getHostIndCount(){return h_indCount;}	
	
	int32_t* getDeviceSrcArray(){return d_edgeSrc;}	
	int32_t* getDeviceDstArray(){return d_edgeDst;}	
	int32_t* getDeviceIndInCompleteArray(){return d_indIncomplete;}	
	int32_t* getDeviceIndCount(){return d_indCount;}	

	void resetHostIndCount(){h_indCount[0]=0;}
	void resetDeviceIndCount(){
		checkCudaErrors(cudaMemset(d_indCount,0,sizeof(int32_t)));
	}

	void copyHostToDevice();
	void copyDeviceToHost();
private:
	int32_t batchSize;

	int32_t *h_edgeSrc,*h_edgeDst, *h_indIncomplete, *h_indCount;
	int32_t *d_edgeSrc,*d_edgeDst, *d_indIncomplete, *d_indCount;

};

