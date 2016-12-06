#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <unordered_map>
#include <algorithm>

#include "utils.hpp"
#include "update.hpp"
#include "memoryManager.hpp"
#include "cuStinger.hpp"


using namespace std;

// As this function uses a hash map it needs to be placed in a .cpp file.
void cuStinger::reAllocateMemoryAfterSweep1(BatchUpdate &bu, length_t& requireAllocation)
{
	// This function consists of two main phases. 
	// 1) Given the list of edges that could be inserted due to lack of space, this list is reduced 
	// such to a unique set of source vertices (i.e. there might be numerous edges that could not be inserted
	// that have the same source vertex.). 
	// 2) Reallocate edge-data memory on the device and copy the previous version of the edge-data.

	vertexId_t *tempsrc=bu.getHostBUD()->getSrc(),*incomplete = bu.getHostBUD()->getIndIncomplete();
	length_t incCount = *(bu.getHostBUD()->getIncCount());

	// The following hash-map used for finding duplicate source edges.
	unordered_map <vertexId_t, length_t> h_hmap;

	for (length_t i=0; i<incCount; i++){
		vertexId_t temp = tempsrc[incomplete[i]];
		h_hmap[temp]++;
	}

	// Contains the list of unique src vertices
	vertexId_t* h_requireUpdates=(vertexId_t*)allocHostArray(*(bu.getHostBUD()->getBatchSize()), sizeof(vertexId_t));
	// For each unique vertex contains how many times that unique source vertex was over the maximal limit
	length_t* h_overLimit=(length_t*)allocHostArray(*(bu.getHostBUD()->getBatchSize()), sizeof(length_t));

	// cout << "The first unique vertex is: "<< tempsrc[incomplete[0]] << endl;
	// Extracting the unique source vertices.
	length_t countUnique=0;
	for (length_t i=0; i<incCount; i++){
		vertexId_t temp = tempsrc[incomplete[i]];
		if(h_hmap[temp]!=0){
			h_requireUpdates[countUnique]=temp;
			h_overLimit[countUnique]=h_hmap[temp];
			countUnique++;
			h_hmap[temp]=0; // Once a vertex is extracted, the hash-map is reset to avoid extracting the source vertex multiple times.
		}
	}
	requireAllocation=0;
	if(countUnique>0){
		// Allocate memory to store the vertex data before the update.
		// We need this information, especially the pointers to the older edge lists as these need to deallocated 
		// after data is copied to the newly allocated arrays.
		cuStinger::cusVertexData* oldhVD = new cuStinger::cusVertexData();
		vertexId_t nv = this->getMaxNV();
		oldhVD->hostAllocateMemoryandInitialize(nv,this->getBytesPerVertex());

		// Copy device VD back to the host. Make an additional host copy for de-allocating memory.
		cuStinger::cusVertexData* cushVD = this->getHostVertexData();
		copyArrayDeviceToHost(this->getDeviceVertexDataMemory(),cushVD->mem,nv,this->getBytesPerVertex());
		copyArrayHostToHost(cushVD->mem,oldhVD->mem,nv,this->getBytesPerVertex());

		// Allocate the necessary memory on the device for the older VD (and make a copy of that meta-data)
		cuStinger::cusVertexData* olddVD = (cuStinger::cusVertexData*)allocDeviceArray(1, sizeof(cuStinger::cusVertexData));
		uint8_t* olddedmem = (uint8_t*)allocDeviceArray(nv,this->getBytesPerVertex());
		this->initVertexDataPointers(olddVD,olddedmem);
		copyArrayHostToDevice(oldhVD->mem,olddedmem,nv,this->getBytesPerVertex());

		edgeBlock** newBlocks = (edgeBlock**)allocHostArray(nv, sizeof(edgeBlock*));

		// For each unique vertex allocate new EdgeData
		for (length_t i=0; i<countUnique; i++){
			vertexId_t tempVertex = h_requireUpdates[i];
			length_t newMax = this->getUpdateAllocater()(cushVD->max[tempVertex] ,h_overLimit[i]);
			// cushVD->adj[tempVertex]  	= (cuStinger::cusEdgeData*)allocDeviceArray(1, sizeof(cuStinger::cusEdgeData));
			// cushVD->edMem[tempVertex]	= (uint8_t*)allocDeviceArray(newMax, this->getBytesPerEdge());

			int memSizeOffsetAdj = sizeof(cusEdgeData)/cudaMemManAlignment + cudaMemManAlignment*(sizeof(cusEdgeData)%cudaMemManAlignment>0);
			int memSizeOffsetedMem = cudaMemManAlignment * (int)ceil ((double) (newMax* this->getBytesPerEdge()) /(double)cudaMemManAlignment);

			memAllocInfo mai = cusMemMan->allocateMemoryBlock(memSizeOffsetAdj+ memSizeOffsetedMem,tempVertex);
			cushVD->adj[tempVertex] = (cusEdgeData*)mai.ptr;
			cushVD->edMem[tempVertex] = (uint8_t*)(mai.ptr+memSizeOffsetAdj);
			cushVD->max[tempVertex] 	= newMax;
			newBlocks[tempVertex] = mai.eb;
		}
		requireAllocation=countUnique;
		// Copy the host VD back to STINGER.
		copyArrayHostToDevice(cushVD->mem,this->dedmem,nv,this->getBytesPerVertex());

		// Copy unique vertex sources to device
		vertexId_t * d_requireUpdates = (vertexId_t*) allocDeviceArray(countUnique, sizeof(vertexId_t));
		copyArrayHostToDevice(h_requireUpdates,d_requireUpdates,countUnique,sizeof(vertexId_t));


		// Modify the data structure on the device. This includes copying all the data concurrently on the device.
			copyMultipleAdjacencies(olddVD,d_requireUpdates,countUnique);

		// cudaEvent_t ce_start,ce_stop;
		// start_clock(ce_start, ce_stop);

		// De-allocate older ED that is no longer needed.
		for (length_t i=0; i<countUnique; i++){
			vertexId_t tempVertex = h_requireUpdates[i];
			cusMemMan->removeMemoryBlock(hMemManEB[tempVertex], tempVertex);
			hMemManEB[tempVertex] = newBlocks[tempVertex];

			// freeDeviceArray(oldhVD->edMem[tempVertex]);
			// freeDeviceArray(oldhVD->adj[tempVertex]);
		}
		// cout << "Reallocate time     : " << end_clock(ce_start, ce_stop) << endl;

		freeHostArray(newBlocks);
		// Remove all auxiliary arrays.
		oldhVD->hostFreeMem();
		delete oldhVD;
		freeDeviceArray(olddedmem);
		freeDeviceArray(olddVD);
		freeDeviceArray(d_requireUpdates);
	}
	freeHostArray(h_requireUpdates); 	
	freeHostArray(h_overLimit);
}
