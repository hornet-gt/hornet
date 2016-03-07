#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include "main.h"


__global__ void devUpdates(int32_t nv,int32_t ne,
	int32_tPtr* d_adjArray,int32_t* d_adjSizeUsed,int32_t* d_adjSizeMax,
	int32_t numUpdates, int32_t* d_updatesSrc, int32_t* d_updatesDst)
{
	int32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos<numUpdates){
		int32_t src = d_updatesSrc[pos];
		int32_t dst = d_updatesDst[pos];

		int32_t ret =  atomicAdd(d_adjSizeUsed+src, 1);

		if(ret<d_adjSizeMax[src]){
			d_adjArray[src][ret] = dst;
		}
		else{
			//RUN out of space
			// printf("*");
		}
	}
}

// void update(int32_t nv,int32_t ne,
// 	int32_tPtr* d_adjArray,int32_t* d_adjSizeUsed,int32_t* d_adjSizeMax,
// 	int32_t numUpdates, int32_t* h_updatesSrc, int32_t* h_updatesDst, 
// 	int32_t* d_updatesSrc, int32_t* d_updatesDst)

void update(int32_t nv,int32_t ne,
	int32_tPtr* d_adjArray,int32_t* d_adjSizeUsed,int32_t* d_adjSizeMax,
	BatchUpdate &bu)
{	
	// copyArrayHostToDevice(h_updatesSrc,d_updatesSrc,numUpdates,sizeof(int32_t));
	// copyArrayHostToDevice(h_updatesDst,d_updatesDst,numUpdates,sizeof(int32_t));

	devUpdates<<<bu.getBatchSize() /32,32>>>(nv,ne,d_adjArray,d_adjSizeUsed,d_adjSizeMax,
	bu.getBatchSize(), bu.getDeviceSrcArray(), bu.getDeviceDstArray());

}




BatchUpdate::BatchUpdate(int32_t batchSize_){
	batchSize=batchSize_;

	h_edgeSrc       =  (int32_t*)allocHostArray(batchSize,sizeof(int32_t));
	h_edgeDst       =  (int32_t*)allocHostArray(batchSize,sizeof(int32_t));
	h_indIncomplete =  (int32_t*)allocHostArray(batchSize,sizeof(int32_t));
	h_indCount      =  (int32_t*)allocHostArray(1,sizeof(int32_t));

	d_edgeSrc       =  (int32_t*)allocDeviceArray(batchSize,sizeof(int32_t));
	d_edgeDst       =  (int32_t*)allocDeviceArray(batchSize,sizeof(int32_t));
	d_indIncomplete =  (int32_t*)allocDeviceArray(batchSize,sizeof(int32_t));
	d_indCount      =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));

}

BatchUpdate::~BatchUpdate(){
	freeHostArray(h_edgeSrc);
	freeHostArray(h_edgeDst);
	freeDeviceArray(d_edgeSrc);
	freeDeviceArray(d_edgeDst);
}

void BatchUpdate::copyHostToDevice(){
	copyArrayHostToDevice(h_edgeSrc, d_edgeSrc, batchSize, sizeof(int32_t));
	copyArrayHostToDevice(h_edgeDst, d_edgeDst, batchSize, sizeof(int32_t));
	copyArrayHostToDevice(h_indIncomplete, d_indIncomplete, batchSize, sizeof(int32_t));
	copyArrayHostToDevice(h_indCount, d_indCount, 1, sizeof(int32_t));
}

void BatchUpdate::copyDeviceToHost(){
	copyArrayDeviceToHost(d_edgeSrc, h_edgeSrc, batchSize, sizeof(int32_t));
	copyArrayDeviceToHost(d_edgeDst, h_edgeDst, batchSize, sizeof(int32_t));
	copyArrayDeviceToHost(d_indIncomplete, h_indIncomplete, batchSize, sizeof(int32_t));
	copyArrayDeviceToHost(d_indCount, h_indCount, 1, sizeof(int32_t));

}

void copyArrayDeviceToHost(void* devSrc, void* hostDst, int32_t elements, int32_t eleSize);



