#pragma once

#include <stdint.h>

typedef int8_t vtype_t;
typedef int8_t etype_t;
typedef int32_t vweight_t;
typedef int32_t eweight_t;
typedef int32_t vertexId_t;
typedef int32_t length_t;
typedef int32_t timestamp_t;

typedef length_t (*initAllocator)(length_t);
length_t defaultInitAllocater(length_t elements);
length_t exactInitAllocater(length_t elements);
length_t stingyInitAllocater(length_t elements);


typedef length_t (*updateAllocator)(length_t, length_t);
length_t defaultUpdateAllocater(length_t elements, length_t overLimit);
length_t exactUpdateAllocater(length_t elements, length_t overLimit);
length_t stingyUpdateAllocater(length_t elements, length_t overLimit);

enum cuStingerInitState{
	eInitStateEmpty,
	eInitStateCSR,
	eInitStateEdgeList,
};

class cuStingerConfig{
public:

	cuStingerInitState initState;

	int maxNV = INT_MAX; // maxNV>csrNV

	bool isSemantic = false;  // Use edge types and vertex types
	bool useVWeight = false;
	bool useEWeight = false;

	// CSR data
	vertexId_t  csrNV 			= INT_MAX;
	length_t    csrNe   		= INT_MAX;
	length_t*   csrOff 			= NULL;
	vertexId_t* csrAdj 			= NULL;
	vweight_t*  csrVW 			= NULL;
	eweight_t*  csrEW			= NULL;

	// Edge List
	vertexId_t* elSrc;
	vertexId_t* elDst;
	eweight_t*  elEW;	
	length_t    elLen;

	// Semantic
	// vtype_t* vTypes   		    = NULL; // 2D array (vTypesLen X 2) . First row - IDS of the vertices that require a type. 
	// 				 					//							   Second row - vertex type for a given vertex.
	// length_t vTypesLen  		= INT_MAX;

};

class cuStinger{
public:
	cuStinger(initAllocator iAllocator=defaultInitAllocater,
		updateAllocator uAllocator=defaultUpdateAllocater);
	~cuStinger();

	void initializeCuStinger(cuStingerConfig);


	void initializeCuStinger(length_t nv_,length_t ne_,length_t* off_, vertexId_t* adj_);
	void copyHostToDevice();
	void copyDeviceToHost(); 

	void freecuStinger();

	__device__ __host__ vertexId_t** getDeviceAdj(){return d_adj;}
	__device__ length_t* getDeviceUtilized(){return d_utilized;}
	__device__ length_t* getDeviceMax(){return d_max;}

	cuStinger* devicePtr(){return d_cuStinger;}


	void copyMultipleAdjacencies(vertexId_t** d_newadj, vertexId_t* requireUpdates, length_t requireCount);

	length_t getNumberEdgesAllocated();
	length_t getNumberEdgesUsed();

public:

	vertexId_t nv;
	bool isSemantic, useVWeight, useEWeight;

	int32_t bytesPerEdge,bytesPerVertex;

// Host memory - this is a shallow copy that does not actually contain the adjacency lists themselves.
	vertexId_t **h_adj;
	length_t *h_utilized,*h_max;
	vweight_t *h_vweight;
	vtype_t *h_vtype;

// Device memory
	vertexId_t **d_adj;
	length_t *d_utilized,*d_max;
	vweight_t *d_vweight;
	vtype_t *d_vtype;


	cuStinger* d_cuStinger;

	initAllocator initVertexAllocator;
	updateAllocator updateVertexAllocator;
	void deviceAllocMemory(length_t* off, vertexId_t* adj);

	void internalEmptycuStinger(int NV);

	void internalCSRcuStinger(length_t* off, vertexId_t* adj, length_t ne);

	length_t sumDeviceArray(length_t* arr);
};


#define CUSTINGER_WARNING(W) std::cout << "cuStinge Warning : " << W << std::endl;
#define CUSTINGER_ERROR(E)   std::cerr << "cuStinge Error   : " << E << std::endl;

// TODO:
// * Add option to send a different element allocator.
