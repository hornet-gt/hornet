#ifndef _CU_UPDATE_INCLUDE_H
#define _CU_UPDATE_INCLUDE_H


#include "cuStingerDefs.hpp"


class BatchUpdateData{
public:
	BatchUpdateData(length_t batchSize_, bool isHost_, length_t nv_=0);
	~BatchUpdateData();

	BatchUpdateData* devicePtr(){return dPtr;}

	// Access functions to the fields in the batch update.
	__host__ __device__ vertexId_t* getSrc(){return edgeSrc;}	
	__host__ __device__ vertexId_t* getDst(){return edgeDst;}	
	__host__ __device__ vertexId_t* getIndIncomplete(){return indIncomplete;}	
	__host__ __device__ vertexId_t* getIndDuplicate(){return indDuplicate;}	
	__host__ __device__ length_t* getDupPosBatch(){return dupPosBatch;}	
	// Single element values
	__host__ __device__ length_t* getIncCount(){return incCount;}	
	__host__ __device__ length_t* getBatchSize(){return batchSize;}
	__host__ __device__ length_t* getNumVertices(){return nv;}
	__host__ __device__ length_t* getDuplicateCount(){return dupCount;}	
	// NV sized arrays
	// TODO: rename to getOff()
	__host__ __device__ length_t* getOffsets(){return offsets;}
	__host__ __device__ length_t* getvNumDuplicates(){return vNumDuplicates;}

	void resetIncCount();
	void resetDuplicateCount();

	void copyDeviceToHostDupCount(BatchUpdateData &dBUD);
	void copyDeviceToHostIncCount(BatchUpdateData &dBUD);

	bool getisHost(){return isHost;}

	// Functions used to synchronize the device and the host.
	void copyHostToHost  (BatchUpdateData &hBUA);
	void copyHostToDevice(BatchUpdateData &hBUA);
	void copyDeviceToHost(BatchUpdateData &dBUA);

private:
	int8_t* mem;
	int64_t numberBytes;
	bool isHost;

	// Arrays storing the batch update information.
	// Each of these arrays is |batchSize|.
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
	length_t* nv;
	// NV sized arrays
	length_t* offsets; // Actually this one is nv+1
	length_t* vNumDuplicates;

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

	void sortDeviceBUD(const int blockdim);

private:

	BatchUpdate* dPtr;
	BatchUpdateData *hData, *dData;
};

#endif