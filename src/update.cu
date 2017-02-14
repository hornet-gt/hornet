#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <kernel_segsort.hxx>

#include "utils.hpp"
#include "update.hpp"

using namespace mgpu;

/// Create a histogram (distribution count) for the source vertex in the batch update
__global__ void calcEdgelistLengths(BatchUpdateData *bud, length_t* const __restrict__ ell){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	length_t batchSize = *(bud->getBatchSize());
	if (tid < batchSize) {
		vertexId_t src = bud->getSrc()[tid];
		atomicAdd(ell+src, 1);
	}
}

/// Copy the edges from the batch update into a CSR graph.
__global__ void copyIndices(BatchUpdateData *bud, vertexId_t* const __restrict__ ind,
	vertexId_t* const __restrict__ seg,	length_t* const __restrict__ off,
	length_t* const __restrict__ ell){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	length_t batchSize = *(bud->getBatchSize());
	if (tid < batchSize)
	{
		vertexId_t src = bud->getSrc()[tid];
		// Start filling up from the end of the edge list like so:
		// ind = ...___|_,_,_,_,_,_,_,3,8,6|_,_,_,_...
		//                el_mark = ^
		length_t el_mark = atomicSub(ell + src, 1) - 1;
		ind[off[src]+el_mark] = bud->getDst()[tid];
		seg[off[src]+el_mark] = src;
	}
}

template <typename T>
__global__ void initDeviceArray(T* mem, int32_t size, T value)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		mem[idx] = value;
	}
}


/// Sort one adjacency list with a single thread.
__device__ void isort(vertexId_t* const __restrict__ u, length_t ell) {
	vertexId_t *v;
	vertexId_t w;
	for (int i = 0; i < ell; ++i) {
		v = u+i;
		while (v != u && *v < *(v-1)) {
			w = *v;
			*v = *(v-1);
			*(v-1) = w;
			v--;
		}
	}
}

/// Sort all the adjacency lists
__global__ void iSortAll(vertexId_t* const __restrict__ ind,
	length_t* const __restrict__ off, length_t nv) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nv) {
		isort( &ind[ off[tid] ], off[tid+1] - off[tid]);
	}
}


void testSort(length_t nv, BatchUpdate& bu,	const int blockdim){

	cudaEvent_t ce_start,ce_stop;
	length_t batchsize = *(bu.getHostBUD()->getBatchSize());

	dim3 numBlocks(1, 1);

	// iSort approach =============================================
	start_clock(ce_start, ce_stop);
	vertexId_t* d_bind = (vertexId_t*) allocDeviceArray(batchsize, sizeof(vertexId_t));
	vertexId_t* d_bseg = (vertexId_t*) allocDeviceArray(batchsize, sizeof(vertexId_t));
	length_t* d_boff = (length_t*) allocDeviceArray(nv+1, sizeof(length_t));
	length_t* d_ell = (length_t*) allocDeviceArray(nv+1, sizeof(length_t));

	numBlocks.x = ceil((float)nv/(float)blockdim);
	initDeviceArray<<<numBlocks,blockdim>>>(d_ell, nv, 0);

	numBlocks.x = ceil((float)batchsize/(float)blockdim);
	calcEdgelistLengths<<<numBlocks,blockdim>>>(bu.getDeviceBUD()->devicePtr(), d_ell);

	thrust::device_ptr<vertexId_t> dp_ell(d_ell);
	thrust::device_ptr<vertexId_t> dp_boff(d_boff);
	thrust::exclusive_scan(dp_ell, dp_ell+nv+1, dp_boff);

	copyIndices<<<numBlocks,blockdim>>>(bu.getDeviceBUD()->devicePtr(), d_bind, d_bseg, d_boff, d_ell);

	numBlocks.x = ceil((float)nv/(float)blockdim);
	iSortAll<<<numBlocks,blockdim>>>(d_bind, d_boff, nv);
	printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));


	// MGPU segsort approach ========================================
	start_clock(ce_start, ce_stop);


	// mgpu::segmented_sort(d_bind, batchsize, d_boff+1, nv-2, mgpu::less_t<int>(), context);

	printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));


	// Thrust approach =============================================
	start_clock(ce_start, ce_stop);

	thrust::device_ptr<vertexId_t> dp_bind(bu.getDeviceBUD()->getDst());
	thrust::device_ptr<vertexId_t> dp_bseg(bu.getDeviceBUD()->getSrc());
	thrust::stable_sort_by_key(dp_bind, dp_bind + batchsize, dp_bseg);
	thrust::stable_sort_by_key(dp_bseg, dp_bseg + batchsize, dp_bind);	

	length_t* d_tell = (length_t*) allocDeviceArray(nv+1, sizeof(length_t));
	length_t* d_tboff = (length_t*) allocDeviceArray(nv+1, sizeof(length_t));

	numBlocks.x = ceil((float)nv/(float)blockdim);
	initDeviceArray<<<numBlocks,blockdim>>>(d_tell, nv, 0);

	numBlocks.x = ceil((float)batchsize/(float)blockdim);
	calcEdgelistLengths<<<numBlocks,blockdim>>>(bu.getDeviceBUD()->devicePtr(), d_tell);

	thrust::device_ptr<vertexId_t> dp_tell(d_tell);
	thrust::device_ptr<vertexId_t> dp_tboff(d_tboff);
	thrust::exclusive_scan(dp_tell, dp_tell+nv+1, dp_tboff);
	printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));


	// Correctness ==============================================

	// From iSort 
	vertexId_t* h_bind = (vertexId_t*) allocHostArray(batchsize, sizeof(vertexId_t));
	vertexId_t* h_bseg = (vertexId_t*) allocHostArray(batchsize, sizeof(vertexId_t));
	length_t* h_boff = (length_t*) allocHostArray(nv+1, sizeof(length_t));

	copyArrayDeviceToHost(d_bind, h_bind, batchsize, sizeof(vertexId_t));
	copyArrayDeviceToHost(d_bseg, h_bseg, batchsize, sizeof(vertexId_t));
	copyArrayDeviceToHost(d_boff, h_boff, nv, sizeof(length_t));

	// From Thrust
	vertexId_t* h_tbind = (vertexId_t*) allocHostArray(batchsize, sizeof(vertexId_t));
	vertexId_t* h_tbseg = (vertexId_t*) allocHostArray(batchsize, sizeof(vertexId_t));
	length_t* h_tboff = (length_t*) allocHostArray(nv+1, sizeof(length_t));

	copyArrayDeviceToHost(bu.getDeviceBUD()->getDst(), h_tbind, batchsize, sizeof(vertexId_t));
	copyArrayDeviceToHost(bu.getDeviceBUD()->getSrc(), h_tbseg, batchsize, sizeof(vertexId_t));
	copyArrayDeviceToHost(d_tboff, h_tboff, nv, sizeof(length_t));

	// Compare
	for (int i = 0; i < nv; ++i)
	{
		if (h_tboff[i] != h_boff[i])
		{
			printf("h_tboff = %d\t h_boff = %d\n", h_tboff[i], h_boff[i]);
		}
	}

	for (int i = 0; i < batchsize; ++i)
	{
		if (h_tbseg[i] != h_bseg[i])
		{
			printf("h_tbseg = %d\t h_bseg = %d\n", h_tbseg[i], h_bseg[i]);
		}
		if (h_tbind[i] != h_bind[i])
		{
			printf("h_tbind = %d\t h_bind = %d\n", h_tbind[i], h_bind[i]);
		}
	}
}

void testmgpusort(){
	mgpu::standard_context_t context;

	int count = 1000;
      int num_segments = div_up(count, 100);
      mem_t<int> segs = fill_random(0, count - 1, num_segments, true, context);
      std::vector<int> segs_host = from_mem(segs);
      mem_t<int> data = fill_random(0, 100000, count, false, context);
      mem_t<int> values(count, context);
      std::vector<int> host_data = from_mem(data);
      segmented_sort(data.data(), count, segs.data(), num_segments,
        less_t<int>(), context);
}

// TODO: change this into a CUDA mem copy operation.
__global__ void copyCSRToBUD(BatchUpdateData *bud, vertexId_t* const __restrict__ ind,
	vertexId_t* const __restrict__ seg)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	length_t batchSize = *(bud->getBatchSize());
	if (tid < batchSize)
	{
		bud->getSrc()[tid] = seg[tid];
		bud->getDst()[tid] = ind[tid];
	}
}

__global__ void copyOffCSRToBUD(BatchUpdateData *bud, length_t* const __restrict__ off)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	length_t nv = *(bud->getNumVertices());
	if (tid < nv+1)
	{
		bud->getOffsets()[tid] = off[tid];
	}
}

void BatchUpdate::sortDeviceBUD(const int blockdim)
{
	length_t batchsize = *(getHostBUD()->getBatchSize());
	length_t nv = *(getHostBUD()->getNumVertices());
	printf("batchsize %d\n", batchsize);

	dim3 numBlocks(1, 1);

	vertexId_t* d_bind = (vertexId_t*) allocDeviceArray(batchsize, sizeof(vertexId_t));
	length_t* d_boff = (length_t*) allocDeviceArray(nv+1, sizeof(length_t));
	vertexId_t* d_bseg = (vertexId_t*) allocDeviceArray(batchsize, sizeof(vertexId_t));
	length_t* d_ell = (length_t*) allocDeviceArray(nv+1, sizeof(length_t));

	numBlocks.x = ceil((float)nv/(float)blockdim);
	// TODO: use memset instead of this hack
	initDeviceArray<<<numBlocks,blockdim>>>(d_ell, nv, 0);
	initDeviceArray<<<numBlocks,blockdim>>>(d_boff, nv, 0);
	// TODO: find a home for this poor statement
	initDeviceArray<<<numBlocks,blockdim>>>(getDeviceBUD()->getvNumDuplicates(), nv, 0);

	numBlocks.x = ceil((float)batchsize/(float)blockdim);
	calcEdgelistLengths<<<numBlocks,blockdim>>>(getDeviceBUD()->devicePtr(), d_ell);

	thrust::device_ptr<vertexId_t> dp_ell(d_ell);
	thrust::device_ptr<vertexId_t> dp_boff(d_boff);
	thrust::exclusive_scan(dp_ell, dp_ell+nv+1, dp_boff);

	copyIndices<<<numBlocks,blockdim>>>(getDeviceBUD()->devicePtr(), d_bind, d_bseg, d_boff, d_ell);

	numBlocks.x = ceil((float)nv/(float)blockdim);
	iSortAll<<<numBlocks,blockdim>>>(d_bind, d_boff, nv);

	// Put the sorted csr back into bud
	numBlocks.x = ceil((float)batchsize/(float)blockdim);
	copyCSRToBUD<<<numBlocks,blockdim>>>(getDeviceBUD()->devicePtr(), d_bind, d_bseg);

	numBlocks.x = ceil((float)(nv+1)/(float)blockdim);
	copyOffCSRToBUD<<<numBlocks,blockdim>>>(getDeviceBUD()->devicePtr(), d_boff);

	freeDeviceArray(d_bind);
	freeDeviceArray(d_boff);
	freeDeviceArray(d_bseg);
	freeDeviceArray(d_ell);
}

