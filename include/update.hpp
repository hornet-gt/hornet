#ifndef _CU_UPDATE_INCLUDE_H
#define _CU_UPDATE_INCLUDE_H


#include "cuStingerDefs.hpp"


class BatchUpdateData{
public:
	BatchUpdateData(length_t batchSize_, bool isHost_);
	~BatchUpdateData();

	BatchUpdateData* devicePtr(){return dPtr;}

	__host__ __device__ vertexId_t* getSrc(){return edgeSrc;}	
	__host__ __device__ vertexId_t* getDst(){return edgeDst;}	
	__host__ __device__ vertexId_t* getIndIncomplete(){return indIncomplete;}	
	__host__ __device__ vertexId_t* getIndDuplicate(){return indDuplicate;}	
	__host__ __device__ length_t* getDupPosBatch(){return dupPosBatch;}	
	// Single element values
	__host__ __device__ length_t* getIncCount(){return incCount;}	
	__host__ __device__ length_t* getBatchSize(){return batchSize;}
	__host__ __device__ length_t* getDuplicateCount(){return dupCount;}	

	void resetIncCount();
	void resetDuplicateCount();

	void copyDeviceToHostDupCount(BatchUpdateData &dBUD);
	void copyDeviceToHostIncCount(BatchUpdateData &dBUD);

	bool getisHost(){return isHost;}

	void copyHostToHost  (BatchUpdateData &hBUA);
	void copyHostToDevice(BatchUpdateData &hBUA);
	void copyDeviceToHost(BatchUpdateData &dBUA);

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
	BatchUpdate(BatchUpdateData &h_bua);
	~BatchUpdate();



	BatchUpdate* devicePtr(){return dPtr;}

	void copyHostToDevice();
	void copyDeviceToHost();

	BatchUpdateData* getHostBUD(){return hData;}
	BatchUpdateData* getDeviceBUD(){return dData;}

private:

	BatchUpdate* dPtr;
	BatchUpdateData *hData, *dData;
};

#endif