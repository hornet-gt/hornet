
#include <stdlib.h>
#include <cuda.h>
// #include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>

#include "cct.hpp"

#include "main.hpp"
// #include "update.hpp"
// #include "cuStinger.hpp"
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

void testSort(length_t nv, BatchUpdate& bu, const int blockdim);
void testmgpusort();

//RNG using Lehmer's Algorithm
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
		printf("%d %d\n", edgeSrc[e], edgeDst[e]);
	}
}

typedef struct dxor128_env {
  unsigned x,y,z,w;
} dxor128_env_t;


// double dxor128(dxor128_env_t * e);
// void dxor128_init(dxor128_env_t * e);
// void dxor128_seed(dxor128_env_t * e, unsigned seed);
void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D, dxor128_env_t * env);

void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D, dxor128_env_t * env){
	int64_t src,dst;
	int scale = (int)log2(double(nv));
	for(int32_t e=0; e<numEdges; e++){
		rmat_edge(&src,&dst,scale, A,B,C,D,env);
		edgeSrc[e] = src;
		edgeDst[e] = dst;
	}
}


int main(const int argc, char *argv[])
{
	testmgpusort();
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	init_timer();
 
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

	// ###                 #
	//  #                  #
	//  #     ##    ###   ###
	//  #    # ##  ##      #
	//  #    ##      ##    #
	//  #     ##   ###      ##
	
	//                                                  #
	//                                                  #
	// ###    ##   #  #         ##    ##   #  #  ###   ###
	// #  #  # ##  #  #        #     #  #  #  #  #  #   #
	// #  #  ##    ####        #     #  #  #  #  #  #   #
	// #  #   ##   ####         ##    ##    ###  #  #    ##
	// =========================================================================
	/*{
		tic();
		// Making a new batch of removed edges
		length_t numEdgesL = numEdges; // Number of edges to remove
		BatchUpdateData bud1(numEdgesL*2,true);
		vertexId_t *src = bud1.getSrc();
		vertexId_t *dst = bud1.getDst();

		// Convert offset arrays to edge length array to update lengths
		length_t *len = (length_t *) malloc (sizeof(length_t)*(nv));
		for(unsigned i = 0; i < nv; ++i) {
			len[i] = off[i+1] - off[i];
		}
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, toc());

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

		// BatchUpdate to add the reverse of the edges
		for(unsigned i = 0; i < numEdgesL; ++i) {
			dst[numEdgesL+i] = src[i];
			src[numEdgesL+i] = dst[i];
		}

		// Count them triangles now
		triangle_t *d_triangles_t = NULL;
		CUDA(cudaMalloc(&d_triangles_t, sizeof(triangle_t)*(nv+1)));
		triangle_t* h_triangles_t = (triangle_t *) malloc (sizeof(triangle_t)*(nv+1));	
		initHostTriangleArray(h_triangles_t,nv);

		tic();
		CUDA(cudaMemcpy(d_triangles_t, h_triangles_t, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));
		callDeviceAllTriangles(custingTest, d_triangles_t, tsp,nbl,shifter,blocks, sps);
		CUDA(cudaMemcpy(h_triangles_t, d_triangles_t, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, toc());

		// Insert them edges now
		BatchUpdate bu1(bud1);
		tic();
		length_t allocs;
		start_clock(ce_start, ce_stop);
		custingTest.edgeInsertions(bu1,allocs);
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, toc());
		printf("%s <%d> %f\n", __FUNCTION__, __LINE__, end_clock(ce_start, ce_stop));

		tic();
		// Sort the new edges
		vertexModification(bu1, nv, custingTest);
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, toc());

		// Count the new triangles now
		triangle_t *d_triangles_new_t = NULL;
		CUDA(cudaMalloc(&d_triangles_new_t, sizeof(triangle_t)*(nv+1)));
		triangle_t* h_triangles_new_t = (triangle_t *) malloc (sizeof(triangle_t)*(nv+1));	
		initHostTriangleArray(h_triangles_new_t,nv);

		tic();
		CUDA(cudaMemcpy(d_triangles_new_t, h_triangles_new_t, sizeof(triangle_t)*(nv+1), cudaMemcpyHostToDevice));
		callDeviceNewTriangles(custingTest, bu1, d_triangles_new_t, tsp,nbl,shifter,blocks, sps, h_triangles, h_triangles_t);
		CUDA(cudaMemcpy(h_triangles_new_t, d_triangles_new_t, sizeof(triangle_t)*(nv+1), cudaMemcpyDeviceToHost));
		printf("\n%s <%d> %f\n", __FUNCTION__, __LINE__, toc());

		// Let's compare
		int64_t sumDevice_t = sumTriangleArray(h_triangles_t,nv);
		int64_t sumDevice_new_t = sumTriangleArray(h_triangles_new_t,nv);
		printf("old %ld, new %ld\n", sumDevice, sumDevice_t);
		printf("============ should be %ld\n", (sumDevice - sumDevice_t)/3);
	}*/
	// =========================================================================


	length_t numEdgesL = numEdges;
	BatchUpdateData bud(numEdgesL,true);
	if(isRmat){
		double a = 0.55, b = 0.15, c = 0.15,d = 0.25;
		dxor128_env_t env;// dxor128_seed(&env, 0);
		generateEdgeUpdatesRMAT(nv, numEdges, bud.getSrc(),bud.getDst(),a,b,c,d,&env);
	}
	else{	
		generateEdgeUpdates(nv, numEdges, bud.getSrc(),bud.getDst());
	}
	BatchUpdate bu(bud);

	testSort(nv, bu, sps);

	cout << endl;

    return 0;	
}       



void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D, dxor128_env_t * env)
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


double dxor128(dxor128_env_t * e) {
  unsigned t=e->x^(e->x<<11);
  e->x=e->y; e->y=e->z; e->z=e->w; e->w=(e->w^(e->w>>19))^(t^(t>>8));
  return e->w*(1.0/4294967296.0);
}

void dxor128_init(dxor128_env_t * e) {
  e->x=123456789;
  e->y=362436069;
  e->z=521288629;
  e->w=88675123;
}

void dxor128_seed(dxor128_env_t * e, unsigned seed) {
  e->x=123456789;
  e->y=362436069;
  e->z=521288629;
  e->w=seed;
}
