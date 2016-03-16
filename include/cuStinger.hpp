#pragma once


typedef int32_t (*initAllocator)(int32_t);
int32_t defaultInitAllocater(int32_t elements);

typedef int32_t (*updateAllocator)(int32_t, int32_t);
int32_t defaultUpdateAllocater(int32_t elements, int32_t overLimit);

class cuStinger{
public:
	cuStinger(initAllocator iAllocator=defaultInitAllocater,
		updateAllocator uAllocator=defaultUpdateAllocater);
	~cuStinger();

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


	int nv;
	int ne;

// Host memory - this is a shallow copy that does not actually contain the adjacency lists themselves.

// Device memory
	int32_t **d_adj,*d_utilized,*d_max;
	int32_t **h_adj,*h_utilized,*h_max;

	cuStinger* d_cuStinger;

	initAllocator initVertexAllocator;
	updateAllocator updateVertexAllocator;
	void deviceAllocMemory(int32_t* off, int32_t* adj);
	void initcuStinger(int32_t* off, int32_t* adj);

	int32_t sumDeviceArray(int32_t* arr);
};


// TODO:
// * Add option to send a different element allocator.
