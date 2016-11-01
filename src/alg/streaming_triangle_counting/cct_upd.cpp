#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include <inttypes.h>
#include <vector>

#include <math.h>

#include "cct.hpp"

#include "main.hpp"
#include "modified.hpp"

using namespace std;

#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)

// Replacement for having another header file
void testmgpusort();
void testSort(length_t nv, BatchUpdate& bu,	const int blockdim);
void callDeviceNewTriangles(cuStinger& custing, BatchUpdate& bu,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim,
    triangle_t * const __restrict__ h_triangles, triangle_t * const __restrict__ h_triangles_t);

void compareCUS(cuStinger* cus1, cuStinger* cus2);


// RNG using Lehmer's Algorithm ================================================
#define RNG_A 16807
#define RNG_M 2147483647
#define RNG_Q 127773
#define RNG_R 2836
#define RNG_SCALE (1.0 / RNG_M)

// Seed can always be changed manually
static int seed = 1;
double getRand(){
    
    int k = seed / RNG_Q;
    seed = RNG_A * (seed - k * RNG_Q) - k * RNG_R;
    
    if (seed < 0) {
        seed += RNG_M;
    }
    
    return seed * (double) RNG_SCALE;
}

void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
	for(int32_t e=0; e<numEdges; e++){
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;
	}
}

void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D)
{
  int64_t i = 0, j = 0;
  int64_t bit = ((int64_t) 1) << (SCALE - 1);

  while (1) {
    const double r =  ((double) rand() / (RAND_MAX));//dxor128(env);
    if (r > A) {                /* outside quadrant 1 */
      if (r <= A + B)           /* in quadrant 2 */
        j |= bit;
      else if (r <= A + B + C)  /* in quadrant 3 */
        i |= bit;
      else {                    /* in quadrant 4 */
        j |= bit;
        i |= bit;
      }
    }
    if (1 == bit)
      break;

    /*
      Assuming R is in (0, 1), 0.95 + 0.1 * R is in (0.95, 1.05).
      So the new probabilities are *not* the old +/- 10% but
      instead the old +/- 5%.
    */
    A *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    B *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    C *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    D *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    /* Used 5 random numbers. */

    {
      const double norm = 1.0 / (A + B + C + D);
      A *= norm;
      B *= norm;
      C *= norm;
    }
    /* So long as +/- are monotonic, ensure a+b+c+d <= 1.0 */
    D = 1.0 - (A + B + C);

    bit >>= 1;
  }
  /* Iterates SCALE times. */
  *iout = i;
  *jout = j;
}

void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D){
	int64_t src,dst;
	int scale = (int)log2(double(nv));
	for(int32_t e=0; e<numEdges; e++){
		rmat_edge(&src,&dst,scale, A,B,C,D);
		edgeSrc[e] = src;
		edgeDst[e] = dst;
	}
}

void printcuStingerUtility(cuStinger custing, bool allInfo){
	length_t used,allocated;

	used     =custing.getNumberEdgesUsed();
	allocated=custing.getNumberEdgesAllocated();
	if (allInfo)
		cout << ", " << used << ", " << allocated << ", " << (float)used/(float)allocated; 	
	else
		cout << ", " << (float)used/(float)allocated;
}

// Search a value in a range of sorted values
template<typename T>
T* search(T* start, int32_t size, T value)
{
	for(unsigned i = 0; i < size; ++i) {
		if(start[i] == value) {
			return start + i;
		}
	}
	return NULL;
}

int SortTest(const int argc, char *argv[])
{
    int isRmat=0;
	int numEdges=10;
	if(argc>1)
		numEdges=atoi(argv[1]);
	if(argc>2)
		isRmat  =atoi(argv[2]);
	srand(100);

	cudaEvent_t ce_start,ce_stop;

	int sps = 128; // Block size
	int nv = 5;

	length_t numEdgesL = numEdges;
	BatchUpdateData bud(numEdgesL,true);
	if(isRmat){
		double a = 0.55, b = 0.15, c = 0.15,d = 0.25;
		generateEdgeUpdatesRMAT(nv, numEdges, bud.getSrc(),bud.getDst(),a,b,c,d);
	}
	else{	
		generateEdgeUpdates(nv, numEdges, bud.getSrc(),bud.getDst());
	}
	BatchUpdate bu(bud);

	testSort(nv, bu, sps);

	cout << endl;

    return 0;
}

void initHostTriangleArray(triangle_t* h_triangles, vertexId_t nv){	
	for(vertexId_t sd=0; sd<(nv);sd++){
		h_triangles[sd]=0;
	}
}

int64_t sumTriangleArray(triangle_t* h_triangles, vertexId_t nv){	
	int64_t sum=0;
	for(vertexId_t sd=0; sd<(nv);sd++){
	  sum+=h_triangles[sd];
	}
	return sum;
}

int InsertionTest(const int argc, char *argv[])
{
	// Making of original custinger
	    length_t nv, ne,*off;
	    vertexId_t *adj;
	    int isRmat=0;
		int numEdges=100000;
		if(argc>2)
			numEdges=atoi(argv[2]);
		if(argc>3)
			isRmat  =atoi(argv[3]);
		srand(100);
		bool isDimacs,isSNAP;
		string filename(argv[1]);
		isDimacs = filename.find(".graph")==std::string::npos?false:true;
		isSNAP   = filename.find(".txt")==std::string::npos?false:true;

		if(isDimacs){
		    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne);
		}
		else if(isSNAP){
		    readGraphSNAP(argv[1],&off,&adj,&nv,&ne);
		}
		else{ 
			cout << "Unknown graph type" << endl;
		}
		cout << nv << ", " << ne;

		cudaEvent_t ce_start,ce_stop;

		cuStinger custing2(defaultInitAllocater,defaultUpdateAllocater);

		cuStingerInitConfig cuInit;
		cuInit.initState =eInitStateCSR;
		cuInit.maxNV = nv+1;
		cuInit.useVWeight = false;
		cuInit.isSemantic = false;  // Use edge types and vertex types
		cuInit.useEWeight = false;

		// CSR data
		cuInit.csrNV 			= nv;
		cuInit.csrNE	   		= ne;
		cuInit.csrOff 			= off;
		cuInit.csrAdj 			= adj;
		cuInit.csrVW 			= NULL;
		cuInit.csrEW			= NULL;

		custing2.initializeCuStinger(cuInit);
		printcuStingerUtility(custing2, false);

	// Counting triangles in original custinger
		triangle_t *d_triangles = NULL;
		CUDA(cudaMalloc(&d_triangles, sizeof(triangle_t)*(nv+1)));
		triangle_t* h_triangles = (triangle_t *) malloc (sizeof(triangle_t)*(nv+1));	
		initHostTriangleArray(h_triangles,nv);

		int tsp = 1; // Threads per intersection
		int shifter = 0; // left shift to multiply threads per intersection
		int sps = 128; // Block size
		int nbl = sps/tsp; // Number of concurrent intersections in block
		int blocks = 16000; // Number of blocks

		tic();
		CUDA(cudaMemcpy(d_triangles, h_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));
		callDeviceAllTriangles(custing2, d_triangles, tsp,nbl,shifter,blocks, sps);
		CUDA(cudaMemcpy(h_triangles, d_triangles, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));
		int64_t sumDevice = sumTriangleArray(h_triangles,nv);
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, toc());
		printf("Triangles in original graph = %ld\n", sumDevice);

	// Make multiple batches of removed edges
		// Convert offset arrays to edge length array to update lengths
		length_t *len = (length_t *) malloc (sizeof(length_t)*(nv));
		for(unsigned i = 0; i < nv; ++i) {
			len[i] = off[i+1] - off[i];
		}

		length_t numEdgesL = numEdges; // Number of edges to remove

		// Number of duplicate entries in batch
		length_t dupEgdes = 1000;
		length_t dupsFromCUS = 1000;
		length_t numTotalEdges = numEdgesL + dupEgdes + dupsFromCUS;

		// Create multiple new batch updates
		int numBatches = 5;
		std::vector<BatchUpdateData*> buds(numBatches);
		for(unsigned i = 0; i < numBatches; ++i) {
			buds[i] = new BatchUpdateData(numTotalEdges*2,true,nv);
		}

		// Make numBatches new batches from the graph
		for(unsigned i = 0; i < numBatches; ++i) {
			BatchUpdateData& budi = *buds[i];
			vertexId_t *src = budi.getSrc();
			vertexId_t *dst = budi.getDst();

			vertexId_t a, b;
			length_t lena;
			// Remove random numEdgesL edges
			for(unsigned i = 0; i < numEdgesL; ++i) {
				do { // Search till you find a not already removed edge
					a = getRand() * nv;
					// new lena used coz len[a] keeps changing when edges are removed
					lena = off[a+1] - off[a];
					if (!lena) continue;
					b = getRand() * lena;
				} while (!lena || adj[ off[a] + b ] == -1);
			
				src[i] = a; dst[i] = adj[ off[a] + b ];
				adj[ off[a] + b ] = -1;
				// Also remove its corresponding entry in dst's edgelist
				*(search(adj+off[dst[i]], off[dst[i]+1] - off[dst[i]], a)) = -1;
				len[a]--; len[dst[i]]--;
			}

			// Add duplicates from beginning of batch
			for(unsigned i = 0; i < dupEgdes; ++i) {
				src[numEdgesL + i] = src[i];
				dst[numEdgesL + i] = dst[i];
			}

			// Get random dupsFromCUS edges
			// (these will be already in cus when applying update)
			for(unsigned i = 0; i < dupsFromCUS; ++i) {
				do { // Search till you find a not already removed edge
					a = getRand() * nv;
					// new lena used coz len[a] keeps changing when edges are removed
					lena = off[a+1] - off[a];
					if (!lena) continue;
					b = getRand() * lena;
				} while (!lena || adj[ off[a] + b ] == -1);
			
				src[numEdgesL + dupEgdes + i] = a;
				dst[numEdgesL + dupEgdes + i] = adj[ off[a] + b ];
			}

			// Per vertex duplicate counter reset
			initHostTriangleArray(budi.getvNumDuplicates(), nv);

			// BatchUpdate to add the reverse of the edges
			for(unsigned i = 0; i < numTotalEdges; ++i) {
				dst[numTotalEdges+i] = src[i];
				src[numTotalEdges+i] = dst[i];
			}
		}

	// Make csr of remaining graph
		// Prefix sum of len array for new offset array
		length_t *newOff = (length_t *) malloc (sizeof(length_t)*(nv+1));
		length_t sum = 0;
		for(unsigned i = 0; i < nv+1; ++i) {
			newOff[i] = sum;
			sum += len[i];
		}
		vertexId_t *newAdj = (vertexId_t *) malloc (sizeof(vertexId_t)*(newOff[nv]));

		// Populate newAdj
		for(unsigned i = 0, j = 0; i < ne; ++i) {
			if (adj[i] != -1) newAdj[j++] = adj[i];
		}

	// Make new custing with newAdj and newOff
		cuInit.csrNE	   		= newOff[nv];
		cuInit.csrOff 			= newOff;
		cuInit.csrAdj 			= newAdj;

		cuStinger custingTest(defaultInitAllocater,defaultUpdateAllocater);
		custingTest.initializeCuStinger(cuInit);

	// Count them triangles now (in new graph with some edges removed)
		// Need to make new arrays so that individual triangle counts can be compared
		triangle_t *d_triangles_t = NULL;
		CUDA(cudaMalloc(&d_triangles_t, sizeof(triangle_t)*(nv+1)));
		triangle_t* h_triangles_t = (triangle_t *) malloc (sizeof(triangle_t)*(nv+1));	
		initHostTriangleArray(h_triangles_t,nv);

		tic();
		CUDA(cudaMemcpy(d_triangles_t, h_triangles_t, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));
		callDeviceAllTriangles(custingTest, d_triangles_t, tsp,nbl,shifter,blocks, sps);
		CUDA(cudaMemcpy(h_triangles_t, d_triangles_t, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, toc());
		int64_t sumDevice_reduced = sumTriangleArray(h_triangles_t,nv);
		printf("Triangles in new reduced graph = %ld\n", sumDevice_reduced);

	// Insert and count, insert and count
		// Need to make new arrays so that individual triangle counts can be compared
		triangle_t *d_triangles_new_t = NULL;
		CUDA(cudaMalloc(&d_triangles_new_t, sizeof(triangle_t)*(nv+1)));
		triangle_t* h_triangles_new_t = (triangle_t *) malloc (sizeof(triangle_t)*(nv+1));	
		initHostTriangleArray(h_triangles_new_t,nv);
		CUDA(cudaMemcpy(d_triangles_new_t, h_triangles_new_t, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));

		for(unsigned i = 0; i < numBatches; ++i) {
			BatchUpdate bu1(*buds[i]);
			length_t allocs;
			
			tic();
			bu1.sortDeviceBUD(sps);
			printf("\n%s <%d> (bud sort) %f\n", __FUNCTION__, __LINE__, toc());
			tic();
			custingTest.edgeInsertionsSorted(bu1, allocs);
			printf("\n%s <%d> (edge insertions) %f\n", __FUNCTION__, __LINE__, toc());


			tic();
			callDeviceNewTriangles(custingTest, bu1, d_triangles_new_t, tsp,nbl,shifter,blocks, sps, h_triangles, h_triangles_t);
			printf("\n%s <%d> (count triangles) %f\n", __FUNCTION__, __LINE__, toc());
		}

		// Add up the triangles and compare to the original graph
		CUDA(cudaMemcpy(h_triangles_new_t, d_triangles_new_t, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));
		int64_t sumDevice_new = sumTriangleArray(h_triangles_new_t,nv);
		printf("Triangles created by batch update = %ld\n", sumDevice_new);

	// Let's compare
	printf("============ should be %ld\n", (sumDevice - sumDevice_reduced));
	printf("old %ld, new %ld\n", sumDevice, sumDevice_reduced);

	custing2.freecuStinger();

	cout << endl;

	free(off);
	free(adj);
    return 0;	

}

int main(const int argc, char *argv[])
{
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	init_timer();

	// Tests ===================================================================
 
	// Test different sorting methods
	// SortTest(argc, argv);

	// Test sorted insertion
	InsertionTest(argc, argv);
}       
