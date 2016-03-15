#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.h"

using namespace std;

// No duplicates allowed

__global__ void deviceUpdatesSweep1(cuStinger* custing, BatchUpdate* bu,int32_t updatesPerBlock)
	// int32_tPtr* d_adj,int32_t* d_utilized,int32_t* d_max,
	// int32_t batchSize, int32_t updatesPerBlock ,i(nt32_t* d_updatesSrc, int32_t* d_updatesDst, 
		// int32_t* d_indIncomplete,int32_t* d_indCount)
{
	int32_t* d_updatesSrc = bu->getDeviceSrc();
	int32_t* d_updatesDst = bu->getDeviceDst();
	int32_t* d_utilized = custing->getDeviceUtilized();
	int32_t* d_max = custing->getDeviceMax();
	int32_t** d_adj = custing->getDeviceAdj();
	int32_t batchSize = bu->getDeviceBatchSize();
	int32_t* d_indCount = bu->getDeviceIndCount();
	int32_t* d_indIncomplete = bu->getDeviceIndIncomplete();

	int32_t* d_indDuplicate = bu->getDeviceIndDuplicate();
	int32_t* d_dupCount = bu->getDeviceDuplicateCount();
	int32_t* d_dupRelPos= bu->getDeviceDupRelPos();


	// if(threadIdx.x==0 && blockIdx.x==0)
	// 	printf("@@@@ %d", batchSize);

	// return;

	int32_t init_pos = blockIdx.x * updatesPerBlock;

	for (int32_t i=0; i<updatesPerBlock; i++){
		int32_t pos=init_pos+i;
		if(pos>=batchSize)
			break;
		int32_t src = d_updatesSrc[pos];
		int32_t dst = d_updatesDst[pos];

		int32_t srcInitSize = d_utilized[src];
		int32_t found=0;
		for (int32_t e=0; e<srcInitSize; e+=blockDim.x){
			if(d_adj[src][e]==dst)
				found=1;
		}
		if(!found && threadIdx.x==0){
			int32_t ret =  atomicAdd(d_utilized+src, 1);
			if(ret<d_max[src]){
				int32_t dupInBatch=0;
				for(int32_t k=srcInitSize; k<ret; k++){
					if (d_adj[src][k]==dst)
						dupInBatch=1;
				}
				if(!dupInBatch){
					d_adj[src][ret] = dst;
				}
				else{
					int32_t duplicateID =  atomicAdd(d_dupCount, 1);
					d_indDuplicate[duplicateID] = pos;
					d_dupRelPos[duplicateID] = ret;
				}
			}
			else{
				// Out of space for this adjacency.
				int32_t inCompleteEdgeID =  atomicAdd(d_indCount, 1);
				d_indIncomplete[inCompleteEdgeID] = pos;
			}
		}
	}
}


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
	deviceUpdatesSweep1<<<numBlocks,threadsPerBlock>>>(custing.devicePtr(), bu.devicePtr(),updatesPerBatch);



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
	h_indDuplicate  =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
	h_dupCount      =  (int32_t*)allocHostArray(1,sizeof(int32_t));
	h_dupRelPos     =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));

	d_edgeSrc       =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
	d_edgeDst       =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
	d_indIncomplete =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
	d_indCount      =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));
	d_batchSize     =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));
	d_indDuplicate  =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
	d_dupCount      =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));
	d_dupRelPos     =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));

	h_batchSize[0]=batchSize_;

	d_batchUpdate=(BatchUpdate*) allocDeviceArray(1,sizeof(BatchUpdate));
	copyArrayHostToDevice(this,d_batchUpdate,1, sizeof(BatchUpdate));
}


BatchUpdate::~BatchUpdate(){
	freeHostArray(h_edgeSrc);
	freeHostArray(h_edgeDst);
	freeHostArray(h_indCount);
	freeHostArray(h_batchSize);
	freeHostArray(h_indDuplicate);
	freeHostArray(h_dupCount);
	freeHostArray(h_dupRelPos);

	freeDeviceArray(d_edgeSrc);
	freeDeviceArray(d_edgeDst);
	freeDeviceArray(d_indCount);
	freeDeviceArray(d_batchSize);
	freeDeviceArray(d_indDuplicate);
	freeDeviceArray(d_dupCount);
	freeDeviceArray(d_dupRelPos);

	freeDeviceArray(d_batchUpdate);

}

void BatchUpdate::copyHostToDevice(){
	copyArrayHostToDevice(h_batchSize, d_batchSize, 1, sizeof(int32_t));

	copyArrayHostToDevice(h_edgeSrc, d_edgeSrc, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_edgeDst, d_edgeDst, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_indIncomplete, d_indIncomplete, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_indCount, d_indCount, 1, sizeof(int32_t));
	copyArrayHostToDevice(h_indDuplicate, d_indDuplicate, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_dupRelPos, d_dupRelPos, h_batchSize[0], sizeof(int32_t));
	copyArrayHostToDevice(h_dupCount, d_dupCount, 1, sizeof(int32_t));
}

void BatchUpdate::copyDeviceToHost(){
	copyArrayDeviceToHost(d_edgeSrc, h_edgeSrc, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_edgeDst, h_edgeDst, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_indIncomplete, h_indIncomplete, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_indCount, h_indCount, 1, sizeof(int32_t));
	copyArrayDeviceToHost(d_indDuplicate, h_indDuplicate, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_dupRelPos, h_dupRelPos, h_batchSize[0], sizeof(int32_t));
	copyArrayDeviceToHost(d_dupCount, h_dupCount, 1, sizeof(int32_t));

}

