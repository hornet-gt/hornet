#include <thrust/device_ptr.h>
#include <thrust/copy.h>

#include "modified.hpp"
#include "utils.hpp"

using namespace std;

__global__ void deviceVertexModification(BatchUpdateData* bud, vertexId_t* d_modV_sparse){
	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();
	length_t batchSize          = *(bud->getBatchSize());

	int32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos>=batchSize)
		return;
	vertexId_t src = d_updatesSrc[pos],dst = d_updatesDst[pos];

	// Adding 1 to the idx which will be subtracted later.
	// This is required to count vertex 0 as modified
	atomicOr(d_modV_sparse + src, src+1);
	atomicOr(d_modV_sparse + dst, dst+1);	
}

template <typename T>
__global__ void setDefault(T* mem, int32_t size, T value)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		mem[idx] = value;
	}
}

struct is_not_zero
{
	__host__ __device__
	bool operator()(const vertexId_t x)
	{
		return (x != 0);
	}
};

void vertexModification(BatchUpdate &bu, length_t nV)
{	
	int32_t threads=1024;	
	dim3 threadsPerBlock(threads, 1);

	dim3 numBlocks(1, 1);
	numBlocks.x = ceil((float)nV/(float)threads);
	vertexId_t* d_modV_sparse = (vertexId_t*)allocDeviceArray(nV, sizeof(vertexId_t));

	// Sets all modified vertices as 0. So size = nV
	setDefault<<<numBlocks,threadsPerBlock>>>(d_modV_sparse, nV, 0);
	checkLastCudaError("Error in vertex modification marking : setting default");

	length_t updateSize = *(bu.getHostBUD()->getBatchSize());
	numBlocks.x = ceil((float)updateSize/(float)threads);
	deviceVertexModification<<<numBlocks,threadsPerBlock>>>(bu.getDeviceBUD()->devicePtr(), d_modV_sparse);
	checkLastCudaError("Error in vertex modification marking : marking modifications");

	vertexId_t* d_modV = (vertexId_t*)allocDeviceArray(updateSize*2, sizeof(vertexId_t));
	thrust::device_ptr<vertexId_t> dp_modV_sparse(d_modV_sparse);
	thrust::device_ptr<vertexId_t> dp_modV(d_modV);
	thrust::copy_if(dp_modV_sparse, dp_modV_sparse + nV, dp_modV, is_not_zero());

	// Testing only. Remove after verified  ====================================

	vertexId_t* h_modV = (vertexId_t*) allocHostArray(updateSize*2, sizeof(vertexId_t));
	copyArrayDeviceToHost(d_modV, h_modV, updateSize*2, sizeof(vertexId_t));

	vertexId_t* modV = (vertexId_t*) allocHostArray(updateSize*2, sizeof(vertexId_t));
	vertexId_t* modV_sparse = (vertexId_t*) allocHostArray(nV, sizeof(vertexId_t));

	for (int i = 0; i < nV; ++i)
	{
		modV_sparse[i] = 0;
	}
	for (int i = 0; i < updateSize; ++i)
	{
		int32_t src = bu.getHostBUD()->getSrc()[i];
		int32_t dst = bu.getHostBUD()->getDst()[i];

		modV_sparse[src] |= (src+1);
		modV_sparse[dst] |= (dst+1);
	}
	int count = 0;
	for (int i = 0; i < updateSize; ++i)
	{
		if (modV_sparse[i])
		{
			modV[count++] = modV_sparse[i];
		}
	}

	for (int i = 0; i < count; ++i)
	{
		if (modV[i] != h_modV[i])
		{
			cout << "error:" << modV[i] << "!=" << h_modV[i];
		}
	}
	// =========================================================================

	checkLastCudaError("Error in vertex modification marking : stream compaction");
}
