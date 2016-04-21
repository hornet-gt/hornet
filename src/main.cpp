
#include <stdlib.h>
#include <cuda.h>
// #include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "main.hpp"
// #include "update.hpp"
// #include "cuStinger.hpp"


using namespace std;



void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
	for(int32_t e=0; e<numEdges; e++){
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;
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
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne,*off;
    vertexId_t *adj;
    int isRmat=0;
	int numEdges=10000;
	if(argc>2)
		numEdges=atoi(argv[2]);
	if(argc>3)
		isRmat  =atoi(argv[3]);
	srand(100);
    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne);

    // cout << argv[1] << endl;
	// cout << "Name : " << prop.name <<  endl;
	cout << "Vertices " << nv << endl;
	cout << "Edges " << ne << endl;

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

	start_clock(ce_start, ce_stop);
	custing2.initializeCuStinger(cuInit);
	cout << "Allocation and Copy Time : " << end_clock(ce_start, ce_stop) << endl;

	cout << "Host utilized   : " << custing2.getNumberEdgesUsed() << endl;
	cout << "Host utilized   : " << custing2.getNumberEdgesAllocated() << endl;

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

	start_clock(ce_start, ce_stop);
		custing2.edgeInsertions(bu);
	cout << "Update time     : " << end_clock(ce_start, ce_stop) << endl;

	custing2.verifyEdgeInsertions(bu);

	cout << "Host utilized   : " << custing2.getNumberEdgesUsed() << endl;
	cout << "Host utilized   : " << custing2.getNumberEdgesAllocated() << endl;

	start_clock(ce_start, ce_stop);
		custing2.edgeDeletions(bu);
	cout << "Update time     : " << end_clock(ce_start, ce_stop) << endl;


	cout << "Host utilized   : " << custing2.getNumberEdgesUsed() << endl;
	cout << "Host utilized   : " << custing2.getNumberEdgesAllocated() << endl;



	cout << "Deletion marker:" << DELETION_MARKER << endl;
	// int cctmain(int nv,int ne, int32_t*  off,int32_t*  ind, cuStinger& custing);
	// cctmain(nv,ne,off,adj,custing2);
	double y=1.9;
	cout << cbrt(y) << endl;

	custing2.freecuStinger();

	free(off);
	free(adj);
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