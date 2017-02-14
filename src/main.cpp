
#include <stdlib.h>
#include <cuda.h>
// #include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "memoryManager.hpp"
#include "cuStinger.hpp"

using namespace std;

/// Printer utility function that gets the percentage of utilized space in the graph.
/// If allInfo==true then it is prints the number of used edges and allocated edges.
void printcuStingerUtility(cuStinger custing, bool allInfo){
	length_t used,allocated;

	used     =custing.getNumberEdgesUsed();
	allocated=custing.getNumberEdgesAllocated();
	if (allInfo)
		cout << "," << used << "," << allocated << "," << (float)used/(float)allocated; 	
	else
		cout << "," << (float)used/(float)allocated;

}

/// Generate an edge list of batch updates using an uniform random edge generator.
void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
	for(int32_t e=0; e<numEdges; e++){
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;
	}
}

typedef struct dxor128_env {
  unsigned x,y,z,w;
} dxor128_env_t;


void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D, dxor128_env_t * env);

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

/// Generate an edge list of batch updates using the RMAT graph random edge generator.
void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D, dxor128_env_t * env){
	int64_t src,dst;
	int scale = (int)log2(double(nv));
	for(int32_t e=0; e<numEdges; e++){
		rmat_edge(&src,&dst,scale, A,B,C,D,env);
		edgeSrc[e] = src;
		edgeDst[e] = dst;
	}
}


/// Example tester for cuSTINGER.
/// Loads an input graph, creates a batches of edges, inserts them into the graph, 
/// and then removes them from the graph.
int main(const int argc, char *argv[])
{
	// Setting the device that will be used.
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne,*off;
    vertexId_t *adj;
    int isRmat=0;

    char* graphName = argv[2];
	// int numBatchEdges=10000;
	srand(100);

	// Reeding the graph.
	bool isDimacs,isSNAP,isMM;
	string filename(argv[1]);
	isDimacs = filename.find(".graph")==std::string::npos?false:true;
	isSNAP   = filename.find(".txt")==std::string::npos?false:true;
	isMM   = filename.find(".mtx")==std::string::npos?false:true;
	isRmat 	 = filename.find("kron")==std::string::npos?false:true;

    bool undirected = hasOption("--undirected", argc, argv);

	if(isDimacs){
	    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne,isRmat);
	}
	else if(isSNAP){
	    readGraphSNAP(argv[1],&off,&adj,&nv,&ne,undirected);
	} else if (isMM) {
		readGraphMatrixMarket(argv[1],&off,&adj,&nv,&ne,undirected);
	}
	else{ 
		cout << "Unknown graph type" << endl;
	}

	cudaEvent_t ce_start,ce_stop;
	cuStingerInitConfig cuInit;
	cuInit.initState =eInitStateCSR;
	cuInit.maxNV = nv+1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;

	// Using a CSR graph for the initial input. 
	cuInit.csrNV 			= nv;
	cuInit.csrNE	   		= ne;
	cuInit.csrOff 			= off;
	cuInit.csrAdj 			= adj;
	cuInit.csrVW 			= NULL;
	cuInit.csrEW			= NULL;

	// Testing the scalablity of edge insertions and deletions for 
	// batch sizes within the range of {1,10,100,.. G_E}
	for (int numBatchEdges=1; numBatchEdges<ne; numBatchEdges*=10){
		// Running each experiment 5 times.
		for (int32_t i=0; i<5; i++){

			cuStinger custing2(defaultInitAllocater,defaultUpdateAllocater);
			start_clock(ce_start, ce_stop);
			custing2.initializeCuStinger(cuInit);
			float initTime = end_clock(ce_start, ce_stop);

			cout << graphName << "," << nv << "," << ne << "," << numBatchEdges;
			cout << "," <<initTime << flush;

			printcuStingerUtility(custing2, false);

			BatchUpdateData bud(numBatchEdges,true);
			// Creating the batch update.
			if(isRmat){ // Using rmat graph generator.
				double a = 0.55, b = 0.15, c = 0.15,d = 0.25;
				dxor128_env_t env;// dxor128_seed(&env, 0);
				generateEdgeUpdatesRMAT(nv, numBatchEdges, bud.getSrc(),bud.getDst(),a,b,c,d,&env);
			}
			else{ // Using a uniform random graph generator.
				generateEdgeUpdates(nv, numBatchEdges, bud.getSrc(),bud.getDst());
			}
			BatchUpdate bu(bud);

			// custing2.checkDuplicateEdges();
			// custing2.verifyEdgeInsertions(bu);
			// cout << "######STARTING INSERTIONS######"<< endl;
			length_t allocs;
			start_clock(ce_start, ce_stop);
				custing2.edgeInsertions(bu,allocs); // Inserting the edges into the graph.
			cout << "," << end_clock(ce_start, ce_stop);
			cout << "," << allocs;

			// custing2.verifyEdgeInsertions(bu);
			// cout << "The graphs are identical" << custing2.verifyEdgeInsertions(bu) << endl;//
			printcuStingerUtility(custing2, false);

			// custing2.checkDuplicateEdges();

			start_clock(ce_start, ce_stop);
				custing2.edgeDeletions(bu); // Inserting the deletions into the graph.
			cout << "," << end_clock(ce_start, ce_stop);
				custing2.verifyEdgeDeletions(bu);
			printcuStingerUtility(custing2, false);
			cout << endl << flush;

			custing2.freecuStinger();

		} 

	}


	free(off);free(adj);
    return 0;	
}       

