#pragma once

#include <stdint.h>

typedef int8_t vtype_t;
typedef int8_t etype_t;
typedef int32_t vweight_t;
typedef int32_t eweight_t;
typedef int32_t vertexId_t;
typedef int32_t length_t;

typedef int32_t (*initAllocator)(int32_t);
int32_t defaultInitAllocater(int32_t elements);
int32_t exactInitAllocater(int32_t elements);
int32_t stingyInitAllocater(int32_t elements);


typedef int32_t (*updateAllocator)(int32_t, int32_t);
int32_t defaultUpdateAllocater(int32_t elements, int32_t overLimit);
int32_t exactUpdateAllocater(int32_t elements, int32_t overLimit);
int32_t stingyUpdateAllocater(int32_t elements, int32_t overLimit);

enum cuStingerInitState{
	eInitStateEmpty,
	eInitStateCSR,
	eInitStateEdgeList,
};

class cuStingerConfig{
public:
	// 
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


	void initializeCuStinger(int32_t nv_,int32_t ne_,int32_t* off_, int32_t* adj_);
	void copyHostToDevice();
	void copyDeviceToHost(); 

	void freecuStinger();

	__device__ __host__ int32_t** getDeviceAdj(){return d_adj;}
	__device__ int32_t* getDeviceUtilized(){return d_utilized;}
	__device__ int32_t* getDeviceMax(){return d_max;}

	cuStinger* devicePtr(){return d_cuStinger;}


	void copyMultipleAdjacencies(int32_t** d_newadj, int32_t* requireUpdates, int32_t requireCount);

	int32_t getNumberEdgesAllocated();
	int32_t getNumberEdgesUsed();

public:

	int nv,nvMax;

// Device memory
	int32_t **d_adj;
	length_t *d_utilized,*d_max;
	vweight_t *d_vweight;
	vtype_t *d_vtype;

// Host memory - this is a shallow copy that does not actually contain the adjacency lists themselves.
	int32_t **h_adj;
	length_t *h_utilized,*h_max;
	vweight_t *h_vweight;
	vtype_t *h_vtype;



	cuStinger* d_cuStinger;

	initAllocator initVertexAllocator;
	updateAllocator updateVertexAllocator;
	void deviceAllocMemory(int32_t* off, int32_t* adj);
	void internalInitcuStinger(int32_t* off, int32_t* adj, int32_t ne);

	int32_t sumDeviceArray(int32_t* arr);
};


// TODO:
// * Add option to send a different element allocator.
