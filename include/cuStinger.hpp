
#ifndef _CU_STINGER_INCLUDE_H
#define _CU_STINGER_INCLUDE_H

#include <stdint.h>

#include "cuStingerDefs.hpp"
class memoryManager;
class edgeBlock;


// The following are the various allocator functions that cuSTINGER supports.
// These are split into two groups. The first group is responsible for deciding 
// on the amount of memory allocated when the graph is created.
// The second group, the update methods, are responsible for the amount of storage
// that will be reallocated whenever a vertex runs out of storage.


typedef length_t (*initAllocator)(length_t);
length_t defaultInitAllocater(length_t elements);
length_t exactInitAllocater(length_t elements);
length_t stingyInitAllocater(length_t elements);


typedef length_t (*updateAllocator)(length_t, length_t);
length_t defaultUpdateAllocater(length_t elements, length_t overLimit);
length_t exactUpdateAllocater(length_t elements, length_t overLimit);
length_t stingyUpdateAllocater(length_t elements, length_t overLimit);

/// These are the various inputs that cuSTINGER currently supports as inputs.
//  In practice, cuSTINGER might take one of the input formats and convert to another
//  format before actually building the graph. CSR is the preferred format.
enum cuStingerInitState{
	eInitStateEmpty,
	eInitStateCSR,
	eInitStateEdgeList,
};

class cuStingerInitConfig{
public:

	cuStingerInitState initState;

	int maxNV = INT_MAX; // maxNV>csrNV
	int defaultBlockSize = 1<<22;

	bool useVWeight = false;

	bool isSemantic = false;  // Use edge types and vertex types
	bool useEWeight = false;

	// CSR data
	vertexId_t  csrNV 			= INT_MAX;
	length_t    csrNE	   		= INT_MAX;
	length_t*   csrOff 			= NULL;
	vertexId_t* csrAdj 			= NULL;
	vweight_t*  csrVW 			= NULL;
	eweight_t*  csrEW			= NULL;

	// Edge List
	vertexId_t* elSrc;
	vertexId_t* elDst;
	eweight_t*  elEW;	
	length_t    elLen;


};


class cuStinger{
public:

	class cusEdgeData{
		friend class cuStinger;
	public:
		uint8_t*		mem;
	public:
		vertexId_t* 	dst;
		eweight_t*   	ew;
		etype_t*    	et;
		timestamp_t*	t1;
		timestamp_t* 	t2;
		__device__ uint8_t* getMem(){return mem;}
	};

	class cusVertexData{
		friend class cuStinger;
	// private:	
	public:
		uint8_t*     mem;
	public:
		cusEdgeData**  	adj;
		uint8_t**      	edMem;
		length_t*    	used;
		length_t*    	max;
		vweight_t*   	vw;
		vtype_t*     	vt;

		void hostAllocateMemoryandInitialize(int nv, int bytesPerVertex){
			mem = (uint8_t*)allocHostArray(nv,bytesPerVertex);
			int32_t pos=0;
			adj 		= (cusEdgeData**)(mem + pos); 	pos+=sizeof(cusEdgeData*)*nv;
			edMem 		= (uint8_t**)(mem + pos); 		pos+=sizeof(uint8_t*)*nv;
			used 		= (length_t*)(mem + pos); 		pos+=sizeof(length_t)*nv;
			max        = (length_t*)(mem + pos); 		pos+=sizeof(length_t)*nv;
			vw         = (vweight_t*)(mem + pos); 		pos+=sizeof(vweight_t)*nv;
			vt         = (vtype_t*)(mem + pos); 		pos+=sizeof(vtype_t)*nv;
		}
		void hostFreeMem(){
			freeHostArray(mem);
		}
// __foreinline__
		__forceinline__ __device__ uint8_t* getMem(){return mem;}
		__forceinline__ __device__ length_t* getUsed(){return used;}
		__forceinline__ __device__ length_t* getMax(){return max;}
		__forceinline__ __device__ cusEdgeData** getAdj(){return adj;}

	};

	cuStinger(initAllocator iAllocator=defaultInitAllocater,
		updateAllocator uAllocator=defaultUpdateAllocater);
	~cuStinger();

	void initializeCuStinger(cuStingerInitConfig &cuInit);
	void initializeCuStinger(length_t nv_,length_t ne_,length_t* off_, vertexId_t* adj_);
	void initVertexDataPointers(cusVertexData *dVD, uint8_t*);
	void initEdgeDataPointers();


	// void copyHostToDevice();
	// void copyDeviceToHost(); 

	void freecuStinger();


	cuStinger* devicePtr(){return d_cuStinger;}


	void copyMultipleAdjacencies(cusVertexData* olddVD, vertexId_t* requireUpdates, length_t requireCount);

	length_t getNumberEdgesAllocated();
	length_t getNumberEdgesUsed();

	inline bool getisSemantic(){return isSemantic;}
	inline bool getuseVWeight(){return useVWeight;}
	inline bool getuseEweight(){return useEWeight;}
	inline __device__ vertexId_t getMaxNV(){return nv;}

	inline updateAllocator getUpdateAllocater(){return updateVertexAllocator;}

	inline length_t getBytesPerVertex(){return bytesPerVertex;}
	inline length_t getBytesPerEdge(){return bytesPerEdge;}

	cusVertexData* getHostVertexData(){return hVD;}
	uint8_t* getDeviceVertexDataMemory(){return dedmem;}

	void edgeInsertions(BatchUpdate &bu, length_t& requireAllocation);
	void edgeInsertionsSorted(BatchUpdate &bu, length_t& requireAllocation);
	void edgeDeletions(BatchUpdate &bu);

	bool verifyEdgeInsertions(BatchUpdate &bu);
	bool verifyEdgeDeletions(BatchUpdate &bu);
	void checkDuplicateEdges();


public:
	vertexId_t nv;
	bool isSemantic, useVWeight, useEWeight;
	int32_t bytesPerEdge,bytesPerVertex;
	cusVertexData *hVD,*dVD;

	cuStinger* d_cuStinger;
	uint8_t* dedmem;

// private: 
	memoryManager* cusMemMan;
	edgeBlock** hMemManEB; 

private:

	initAllocator initVertexAllocator;
	updateAllocator updateVertexAllocator;
	void deviceAllocMemory(length_t* off, vertexId_t* adj);

	void internalEmptyTocuStinger(int NV);
	void internalCSRTocuStinger(length_t* off, vertexId_t* adj, length_t ne);

	length_t sumDeviceArray(length_t* arr, length_t);

	void reAllocateMemoryAfterSweep1(BatchUpdate &bu,length_t& requireAllocation);

};


#endif