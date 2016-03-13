#pragma once


class cuStinger{
public:
	cuStinger(){}
	~cuStinger();

	void hostCsrTocuStinger(int32_t nv_,int32_t ne_,int32_t* off_, int32_t* adj_);


	void copyHostToDevice();

	void freecuStinger();

	__device__ __host__ int32_t** getAdjArray(){return d_adjArray;}
	__device__ int32_t* getSizeUsedArray(){return d_adjSizeUsed;}
	__device__ int32_t* getSizeMaxArray(){return d_adjSizeMax;}

	cuStinger* devicePtr(){return d_cuStinger;}

// Not yet supported	
//	void copyDeviceToHost(); 


public:

	// cuStinger(const cuStinger& custing)
	// :d_adjArray(custing.d_adjArray),
	// d_adjSizeUsed(custing.d_adjSizeUsed),
	// d_adjSizeMax(custing.d_adjSizeMax)
	// {
	// }

	int nv;
	int ne;

// Device memory
	int32_t** d_adjArray;
	int32_t* d_adjSizeUsed;
	int32_t* d_adjSizeMax;

	cuStinger* d_cuStinger;

	void deviceAllocMemory(int32_t* off, int32_t* adj);
	void initcuStinger(int32_t* off, int32_t* adj);
};


	// int32_t** d_adj;
	// int32_t* d_utilized;
	// int32_t* d_max;


// TODO:
// * Add option to send a different element allocator.
