#ifndef _CU_STINGER_MEMORY_MANAGER_H_
#define _CU_STINGER_MEMORY_MANAGER_H_

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <inttypes.h>

#include "stx/btree_multiset.h"
#include "stx/btree_multimap.h"
#include "stx/btree_map.h"

#include "cuStingerDefs.hpp"

using namespace std;



struct memVertexData{
	memVertexData(){}
	memVertexData(vertexId_t vertex_,uint64_t offset_,uint64_t size_){
		vertex=vertex_;
		offset=offset_;
		size=size_;
	}

	vertexId_t 	 vertex;
	uint64_t offset;
	uint64_t size;
};

typedef stx::btree_multimap<vertexId_t, memVertexData,std::less<vertexId_t> > offsetTree;

class edgeBlock
{
public:
	edgeBlock(){}
	edgeBlock(uint64_t blockSize_);
	edgeBlock(const edgeBlock& eb);
	~edgeBlock(){}

	void releaseInnerTree();

	edgeBlock& operator=(const edgeBlock &eb );
	uint64_t addMemBlock(uint64_t memSize,vertexId_t v);
	void removeMemBlock(vertexId_t v);
	uint64_t getAvailableSpace();
	uint8_t* getMemoryBlockPtr();
	edgeBlock* getEdgeBlockPtr();
	uint64_t elementsInEdgeBlock();

private:
	uint8_t* memoryBlockPtr;
	uint64_t blockSize;
	uint64_t utilization;
	uint64_t availbility;
	uint64_t nextAvailable;
	offsetTree* offTree;
};


typedef edgeBlock* edgeBlockPtr;
typedef stx::btree_multimap<uint64_t, edgeBlockPtr, std::less<uint64_t> > ebBPtree;

struct memAllocInfo{
	memAllocInfo(){}
	memAllocInfo(edgeBlock* eb_,uint8_t*  ptr_){
		eb=eb_;
		ptr=ptr_;
	}
	edgeBlock* eb;
	uint8_t*  ptr;

};

const int64_t cudaMemManAlignment = 64;

class memoryManager
{
public:
	memoryManager(uint64_t blockSize_);
	~memoryManager();
	memAllocInfo allocateMemoryBlock(uint64_t memSize,vertexId_t v);
	void removeMemoryBlock(edgeBlock* eb,vertexId_t v);

	uint64_t getTreeSize();

#if(MEM_STAND_ALONE) 
public:
#else
private:
#endif
	ebBPtree btree;
	uint64_t blockSize;
};



#endif

