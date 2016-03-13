#pragma once


class cuStinger{
public:
	cuStinger(){}
	~cuStinger();

	void hostCsrTocuStinger(int32_t nv_,int32_t ne_,int32_t* off_, int32_t* adj_);


	void copyHostToDevice();

	void freecuStinger();

	__device__ __host__ int32_t** getDeviceAdj(){return d_adj;}
	__device__ int32_t* getDeviceUtilized(){return d_utilized;}
	__device__ int32_t* getDeviceMax(){return d_max;}

	cuStinger* devicePtr(){return d_cuStinger;}

// Not yet supported	
//	void copyDeviceToHost(); 


public:

	// cuStinger(const cuStinger& custing)
	// :d_adj(custing.d_adj),
	// d_utilized(custing.d_utilized),
	// d_max(custing.d_max)
	// {
	// }

	int nv;
	int ne;

// Device memory
	int32_t** d_adj;
	int32_t* d_utilized;
	int32_t* d_max;

	cuStinger* d_cuStinger;

	void deviceAllocMemory(int32_t* off, int32_t* adj);
	void initcuStinger(int32_t* off, int32_t* adj);
};


	// int32_t** d_adj;
	// int32_t* d_utilized;
	// int32_t* d_max;


// TODO:
// * Add option to send a different element allocator.
