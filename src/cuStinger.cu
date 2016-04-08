
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "main.hpp"

using namespace std;



// void initializeCuStinger(cuStingerConfig);

__global__ void devInitVertexData(cuStinger* custing,uint8_t* temp)
{
	if(threadIdx.x!=0 || blockIdx.x!=0)
		DEV_CUSTINGER_ERROR("Number of threads and thread blocks for initializing vertex should always be one");
	cuStinger::cusVertexData *dVD = custing->dVD;

	dVD->mem = temp;
	int32_t pos=0;
	int32_t nv = custing->nv;

	// printf("Vertex: From the device : %p \n",dVD);
	// printf("Vertex: From the device : %p \n",temp);

	dVD->adj 		= (cuStinger::cusEdgeData**)(dVD->getMem() + pos); 	pos+=sizeof(cuStinger::cusEdgeData*)*nv;
	dVD->edMem 		= (uint8_t**)(dVD->getMem() + pos); 				pos+=sizeof(uint8_t*)*nv;
	dVD->used 		= (length_t*)(dVD->getMem() + pos); 				pos+=sizeof(length_t)*nv;
	dVD->max        = (length_t*)(dVD->getMem() + pos); 				pos+=sizeof(length_t)*nv;
	dVD->vw         = (vweight_t*)(dVD->getMem() + pos); 				pos+=sizeof(vweight_t)*nv;
	dVD->vt         = (vtype_t*)(dVD->getMem() + pos); 					pos+=sizeof(vtype_t)*nv;
}

void cuStinger::initVertexDataPointers(uint8_t* temp){
	devInitVertexData<<<1,1>>>(	d_cuStinger,temp);
}

__global__ void devInitEdgeData(cuStinger* custing, int verticesPerThreadBlock)
{
	// if(threadIdx.x==0 && blockIdx.x==10)
	// 	printf("The number of vertices is : %d\n", custing->nv);

	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock+threadIdx.x;
	length_t nv = custing->nv;
	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat+=blockDim.x){
		vertexId_t v=v_init+v_hat;
		if(v>=custing->nv)
			break;
		//epv = edge per vertex
		length_t epv = custing->getDeviceMax()[v];
		int32_t pos=0;
		cuStinger::cusEdgeData *dED = custing->dVD->adj[v];

		dED->mem = custing->dVD->edMem[v];

		if(threadIdx.x==0 && blockIdx.x==10) printf("####%d,",epv);
		dED->dst = (vertexId_t*)(dED->getMem() + pos); 	pos+=sizeof(vertexId_t)*epv;
		if(threadIdx.x==0 && blockIdx.x==10) printf("%p,",dED->dst);
		dED->ew  = (eweight_t*)(dED->getMem() + pos); 	pos+=sizeof(eweight_t)*epv;
		if(threadIdx.x==0 && blockIdx.x==10) printf("%d,",pos);
		dED->et  = (etype_t*)(dED->getMem() + pos); 	pos+=sizeof(etype_t)*epv;
		if(threadIdx.x==0 && blockIdx.x==10) printf("%d,",pos);
		dED->t1  = (timestamp_t*)(dED->getMem() + pos); pos+=sizeof(timestamp_t)*epv;
		if(threadIdx.x==0 && blockIdx.x==10) printf("%d,",pos);
		dED->t2  = (timestamp_t*)(dED->getMem() + pos); pos+=sizeof(timestamp_t)*epv;
		if(threadIdx.x==0 && blockIdx.x==10) printf("%d,",pos);
		if(threadIdx.x==0 && blockIdx.x==10) printf("####\n");
	}
}

void cuStinger::initEdgeDataPointers(){
	dim3 numBlocks(1, 1);
	int32_t threads=64;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)nv/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	

	int32_t verticesPerThreadBlock = threads;
	ceil(float(nv)/float(numBlocks.x));
	// if(numBlocks.x>1)
	// 	 verticesPerThreadBlock = ceil(float(nv)/float(numBlocks.x-1));		
	cout << "**** Number of vertices per block " << verticesPerThreadBlock << endl;
	devInitEdgeData<<<numBlocks,threadsPerBlock>>>(	d_cuStinger,verticesPerThreadBlock);
}

__global__ void devMakeGPUStinger(int32_t* d_off, int32_t* d_adj,
	int verticesPerThreadBlock,cuStinger* custing){
	length_t* d_utilized = custing->getDeviceUsed();

	int32_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		int32_t v=v_init+v_hat;
		if(v>=custing->nv)
			break;
		cuStinger::cusEdgeData* adjv = custing->dVD->adj[v];

		for(int32_t e=threadIdx.x; e<d_utilized[v]; e+=blockDim.x){
			// d_cuadj[v][e]=d_adj[d_off[v]+e];
			adjv->dst[e]=d_adj[d_off[v]+e];
			// adj->dst[0]=1;
		}
	}
}


void cuStinger::internalCSRTocuStinger(length_t* h_off, vertexId_t* h_adj, int ne){
	length_t* d_off = (length_t*)allocDeviceArray(nv+1,sizeof(int32_t));
	vertexId_t* d_adj = (length_t*)allocDeviceArray(ne,sizeof(int32_t));
	copyArrayHostToDevice(h_off,d_off,nv,sizeof(length_t));
	copyArrayHostToDevice(h_adj,d_adj,ne,sizeof(vertexId_t));

	dim3 numBlocks(1, 1);
	int32_t threads=64;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)nv/(float)threads);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	

	int32_t verticesPerThreadBlock = ceil(float(nv)/float(numBlocks.x-1));

	devMakeGPUStinger<<<numBlocks,threadsPerBlock>>>(d_off,d_adj,verticesPerThreadBlock, d_cuStinger);

	freeDeviceArray(d_adj);	
	freeDeviceArray(d_off);
}



#define SUM_BLOCK_SIZE 512
__global__ void total(length_t * input, length_t * output, length_t len) {
    __shared__ length_t partialSum[2 * SUM_BLOCK_SIZE];
    //Load a segment of the input vector into shared memory
    length_t tid = threadIdx.x, start = 2 * blockIdx.x * SUM_BLOCK_SIZE;
    if (start + tid < len)
       partialSum[tid] = input[start + tid];
    else
       partialSum[tid] = 0;

    if (start + SUM_BLOCK_SIZE + tid < len)
       partialSum[SUM_BLOCK_SIZE + tid] = input[start + SUM_BLOCK_SIZE + tid];
    else
       partialSum[SUM_BLOCK_SIZE + tid] = 0;

    //Traverse the reduction tree
    for (int stride = SUM_BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (tid < stride)
          partialSum[tid] += partialSum[tid+stride];
    }
    //Write the computed sum of the block to the output vector at the correct index
    if (tid == 0)
       output[blockIdx.x] = partialSum[0];
}


length_t cuStinger::sumDeviceArray(length_t* arr, length_t len){
	length_t numOutputElements = len / (SUM_BLOCK_SIZE<<1);
    if (len % (SUM_BLOCK_SIZE<<1)) {
        numOutputElements++;
    }

	length_t* d_out = (length_t*)allocDeviceArray(len, sizeof(length_t*));

	total<<<numOutputElements,SUM_BLOCK_SIZE>>>(arr,d_out,len);

	length_t* h_out = (int32_t*)allocHostArray(len, sizeof(length_t*));
	
	length_t sum=0;
	copyArrayDeviceToHost(d_out, h_out, len, sizeof(length_t));
	for(int i=0; i<numOutputElements; i++){
		 // cout << h_out[i] << ", ";
		sum+=h_out[i];
	}
	freeHostArray(h_out);
	freeDeviceArray(d_out);	
	return sum;
}



__global__ void deviceCopyMultipleAdjacencies(cuStinger* custing, vertexId_t** d_newadj, 
	vertexId_t* requireUpdates, length_t requireCount ,length_t verticesPerThreadBlock)
{
	// int32_t** d_cuadj = custing->d_adj;
	length_t* d_utilized = custing->getDeviceUsed();

	length_t v_init=blockIdx.x*verticesPerThreadBlock;
	for (int v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		if((v_init+v_hat)>=requireCount)
			break;
		vertexId_t v=requireUpdates[v_init+v_hat];

		for(length_t e=threadIdx.x; e<d_utilized[v]; e+=blockDim.x){
			// d_newadj[v][e] = d_cuadj[v][e];
		}
	}
}

void cuStinger::copyMultipleAdjacencies(vertexId_t** d_newadj, 
	vertexId_t* requireUpdates, length_t requireCount){

	dim3 numBlocks(1, 1);
	int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);

	numBlocks.x = ceil((float)requireCount);
	if (numBlocks.x>16000){
		numBlocks.x=16000;
	}	
	int32_t verticesPerThreadBlock;
	if(numBlocks.x == requireCount)
		verticesPerThreadBlock=1;
	else
		verticesPerThreadBlock = ceil(float(requireCount)/float(numBlocks.x-1));

	cout << "### " << requireCount << " , " <<  numBlocks.x << " , " << verticesPerThreadBlock << " ###"  << endl; 

	deviceCopyMultipleAdjacencies<<<numBlocks,threadsPerBlock>>>(d_cuStinger,
		d_newadj, requireUpdates, requireCount, verticesPerThreadBlock);
	checkLastCudaError("Error in the first update sweep");
}



