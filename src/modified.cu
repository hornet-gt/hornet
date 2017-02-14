#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include "modified.hpp"
#include "utils.hpp"
#include "cuStinger.hpp"
#include "update.hpp"

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

__global__ void GetEdgeLengths(cuStinger* cus, vertexId_t* d_modV, length_t num_modV, length_t* d_mV_edge_l)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num_modV) {
		d_mV_edge_l[tid] = cus->dVD->getUsed()[d_modV[tid]-1];
	}
}

__global__ void CopyEdgeListToScratchpadOBPV(cuStinger* cus, vertexId_t* d_modV,
	length_t scratchpad_l, vertexId_t* d_mV_scratch, vertexId_t* d_mV_segment, length_t* d_mV_edge_l)
{
	vertexId_t vertex = d_modV[blockIdx.x]-1;
	length_t startpoint = (blockIdx.x == 0)?0:d_mV_edge_l[blockIdx.x-1];
	length_t endpoint = d_mV_edge_l[blockIdx.x];

	vertexId_t* e_list = cus->dVD->getAdj()[vertex]->dst;

	for (int i = startpoint+threadIdx.x; i < endpoint; i+=blockDim.x) {
		d_mV_scratch[i] = e_list[i - startpoint];
		d_mV_segment[i] = vertex;
	}
}

__global__ void CopyScratchpadToEdgeListOBPV(cuStinger* cus, vertexId_t* d_modV,
	length_t scratchpad_l, vertexId_t* d_mV_scratch, vertexId_t* d_mV_segment, length_t* d_mV_edge_l)
{
	vertexId_t vertex = d_modV[blockIdx.x]-1;
	length_t startpoint = (blockIdx.x == 0)?0:d_mV_edge_l[blockIdx.x-1];
	length_t endpoint = d_mV_edge_l[blockIdx.x];

	vertexId_t* e_list = cus->dVD->getAdj()[vertex]->dst;

	for (int i = startpoint+threadIdx.x; i < endpoint; i+=blockDim.x) {
		e_list[i - startpoint] = d_mV_scratch[i];
	}
}

void vertexModification(BatchUpdate &bu, length_t nV, cuStinger &cus)
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
	
	cudaEvent_t ce_start,ce_stop;
	start_clock(ce_start, ce_stop);
	vertexId_t* d_modV_end = thrust::raw_pointer_cast(thrust::copy_if(dp_modV_sparse, dp_modV_sparse + nV, dp_modV, is_not_zero()));
	printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));

	checkLastCudaError("Error in vertex modification marking : stream compaction");

	length_t num_modV = d_modV_end - d_modV;

	//  ##                #     #
	// #  #               #
	//  #     ##   ###   ###   ##    ###    ###
	//   #   #  #  #  #   #     #    #  #  #  #
	// #  #  #  #  #      #     #    #  #   ##
	//  ##    ##   #       ##  ###   #  #  #
	//                                      ###
	// =========================================================================

	// Allocate new array of size num_modV
	length_t* d_mV_edge_l = (length_t*)allocDeviceArray(num_modV, sizeof(length_t));
	numBlocks.x = ceil((float)num_modV/(float)threads);
	GetEdgeLengths<<<numBlocks,threadsPerBlock>>>(cus.devicePtr(), d_modV, num_modV, d_mV_edge_l);

	// Testing again
	_DEBUG(
		length_t* h_mV_edge_l = (length_t*) allocHostArray(num_modV, sizeof(length_t));
		copyArrayDeviceToHost(d_mV_edge_l, h_mV_edge_l, num_modV, sizeof(length_t));
	)

	// Vectorized (thrust stable sort) approach
	// =========================================================================
		// Prefix sum for this array
		thrust::device_ptr<vertexId_t> dp_mV_edge_l(d_mV_edge_l);
		thrust::inclusive_scan(dp_mV_edge_l, dp_mV_edge_l+num_modV, dp_mV_edge_l);

		// Testing
		_DEBUG(copyArrayDeviceToHost(d_mV_edge_l, h_mV_edge_l, num_modV, sizeof(length_t));)

		// Allocation of scratchpad
		length_t* scratchpad_l = (length_t*) allocHostArray(1, sizeof(length_t));
		copyArrayDeviceToHost(d_mV_edge_l+num_modV-1, scratchpad_l, 1, sizeof(length_t));
		vertexId_t* d_mV_scratch = (vertexId_t*)allocDeviceArray(scratchpad_l[0], sizeof(vertexId_t));

		// Create segment data
		vertexId_t* d_mV_segment = (vertexId_t*)allocDeviceArray(scratchpad_l[0], sizeof(vertexId_t));

		// Copy edge arrays to scratchpad
		numBlocks.x = ceil((float)scratchpad_l[0]/(float)threads);
		enum COPY_METHOD { ONE_BLOCK_PER_V, ONE_THREAD_PER_E_BIN_SRCH, ONE_THREAD_PER_E_PRFX_SUM };
		COPY_METHOD method = ONE_BLOCK_PER_V;

		CopyEdgeListToScratchpadOBPV<<<num_modV,threadsPerBlock>>>(cus.devicePtr(),
			d_modV, scratchpad_l[0], d_mV_scratch, d_mV_segment, d_mV_edge_l);

		// Testing again
		_DEBUG(
			vertexId_t* h_mV_scratch = (vertexId_t*) allocHostArray(scratchpad_l[0], sizeof(vertexId_t));
			copyArrayDeviceToHost(d_mV_scratch, h_mV_scratch, scratchpad_l[0], sizeof(vertexId_t));
			vertexId_t* h_mV_segment = (vertexId_t*) allocHostArray(scratchpad_l[0], sizeof(vertexId_t));
			copyArrayDeviceToHost(d_mV_segment, h_mV_segment, scratchpad_l[0], sizeof(vertexId_t));
		)

		// Actual Sort operation
		thrust::device_ptr<vertexId_t> dp_mV_scratch(d_mV_scratch);
		thrust::device_ptr<vertexId_t> dp_mV_segment(d_mV_segment);
		thrust::stable_sort_by_key(dp_mV_scratch, dp_mV_scratch + scratchpad_l[0], dp_mV_segment);
		thrust::stable_sort_by_key(dp_mV_segment, dp_mV_segment + scratchpad_l[0], dp_mV_scratch);

		// Testing testing testing
		_DEBUG(copyArrayDeviceToHost(d_mV_scratch, h_mV_scratch, scratchpad_l[0], sizeof(vertexId_t));)

		// Copy back to original location
		CopyScratchpadToEdgeListOBPV<<<num_modV,threadsPerBlock>>>(cus.devicePtr(),
			d_modV, scratchpad_l[0], d_mV_scratch, d_mV_segment, d_mV_edge_l);
}
