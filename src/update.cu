#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include "main.h"

using namespace std;

// __global__ void devUpdates(
// 	int32_tPtr* d_adjArray,int32_t* d_adjSizeUsed,int32_t* d_adjSizeMax,
// 	int32_t batchSize, int32_t updatesPerBlock ,int32_t* d_updatesSrc, int32_t* d_updatesDst, 
// 	int32_t* d_indIncomplete,int32_t* d_indCount)

__global__ void devUpdates(cuStinger* custing, BatchUpdate* bu,int32_t updatesPerBlock)
	// int32_tPtr* d_adjArray,int32_t* d_adjSizeUsed,int32_t* d_adjSizeMax,
	// int32_t batchSize, int32_t updatesPerBlock ,i(nt32_t* d_updatesSrc, int32_t* d_updatesDst, 
		// int32_t* d_indIncomplete,int32_t* d_indCount)
{
	int32_t* d_updatesSrc = bu->getDeviceSrcArray();
	int32_t* d_updatesDst = bu->getDeviceDstArray();
	int32_t* d_adjSizeUsed = custing->getSizeUsedArray();
	int32_t* d_adjSizeMax = custing->getSizeMaxArray();
	int32_t** d_adjArray = custing->getAdjArray();
	int32_t batchSize = bu->getDeviceBatchSize();
	int32_t* d_indCount = bu->getDeviceIndCount();
	int32_t* d_indIncomplete = bu->getDeviceIndInCompleteArray();

	// if(threadIdx.x==0 && blockIdx.x==0)
	// 	printf("@@@@ %d", batchSize);

	// return;
	int32_t init_pos = blockIdx.x * updatesPerBlock;
	for(int i=threadIdx.x; i<updatesPerBlock; i+=blockDim.x){
		int32_t pos=init_pos+i;
		if(pos<batchSize){
			// if (pos<0){
			// 	printf("***** %d %d \n", init_pos, i);
			// 	break;
			// }
			int32_t src = d_updatesSrc[pos];
			int32_t dst = d_updatesDst[pos];
			int32_t ret =  atomicAdd(d_adjSizeUsed+src, 1);

			if(ret<d_adjSizeMax[src]){
				d_adjArray[src][ret] = dst;
			}
			else{
				int32_t inCompleteEdgeID =  atomicAdd(d_indCount, 1);
				d_indIncomplete[inCompleteEdgeID] = pos;
				//RUN out of space
				// printf("%d %d \n", inCompleteEdgeID,pos);
				// printf("*");
			}
			// d_updatesSrc[pos]=0;
			// d_updatesDst[pos]=0;
		}

	}
}

// void update(int32_t nv,int32_t ne,
// 	int32_tPtr* d_adjArray,int32_t* d_adjSizeUsed,int32_t* d_adjSizeMax,
// 	int32_t numUpdates, int32_t* h_updatesSrc, int32_t* h_updatesDst, 
// 	int32_t* d_updatesSrc, int32_t* d_updatesDst)

void update(cuStinger &custing, BatchUpdate &bu)
{	
	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t batchSize = bu.getHostBatchSize();
	numBlocks.x = ceil((float)batchSize/(float)threads);

	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	int32_t updatesPerBatch = ceil(float(batchSize)/float(numBlocks.x-1));

	cout << numBlocks.x << " : " << threadsPerBlock.x << " : " << updatesPerBatch << endl;
	devUpdates<<<numBlocks,threadsPerBlock>>>(custing.devicePtr(), bu.devicePtr(),updatesPerBatch);

	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
		fprintf(stderr,"ERROR1: %s: %s - %d\n", "Update error", cudaGetErrorString(error), error );
	}
}


BatchUpdate::BatchUpdate(int32_t batchSize_){

	h_edgeSrc       =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
	h_edgeDst       =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
	h_indIncomplete =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
	h_indCount      =  (int32_t*)allocHostArray(1,sizeof(int32_t));
	h_batchSize     =  (int32_t*)allocHostArray(1,sizeof(int32_t));
	h_batchSize[0]=batchSize_;

	d_edgeSrc       =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
	d_edgeDst       =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
	d_indIncomplete =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
	d_indCount      =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));
	d_batchSize     =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));

	d_batchUpdate=(BatchUpdate*) allocDeviceArray(1,sizeof(BatchUpdate));
	copyArrayHostToDevice(this,d_batchUpdate,1, sizeof(BatchUpdate));
}


BatchUpdate::~BatchUpdate(){
	freeHostArray(h_edgeSrc);
	freeHostArray(h_edgeDst);
	freeDeviceArray(d_edgeSrc);
	freeDeviceArray(d_edgeDst);
	freeDeviceArray(d_batchUpdate);
	freeDeviceArray(d_indCount);
	freeDeviceArray(d_batchSize);
}

void BatchUpdate::copyHostToDevice(){
	copyArrayHostToDevice(h_edgeSrc, d_edgeSrc, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_edgeDst, d_edgeDst, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_indIncomplete, d_indIncomplete, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_indCount, d_indCount, 1, sizeof(int32_t));
	copyArrayHostToDevice(h_batchSize, d_batchSize, 1, sizeof(int32_t));

	cout << "Batch size is " << h_batchSize[0] << endl;
}

void BatchUpdate::copyDeviceToHost(){
	copyArrayDeviceToHost(d_edgeSrc, h_edgeSrc, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_edgeDst, h_edgeDst, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_indIncomplete, h_indIncomplete, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_indCount, h_indCount, 1, sizeof(int32_t));
}

