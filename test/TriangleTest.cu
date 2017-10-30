
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include "Static/TriangleCounting/triangle.cuh"
#include "Device/Timer.cuh"

#include <GraphIO/GraphStd.hpp>

using namespace std;
using namespace timer;
using namespace hornets_nest;

#define STAND_PRINTF(sys, time, triangles) printf("%s : \t%ld \t%f\n", sys,triangles, time);

// int arrayBlocks[]={16000};
// int arrayBlockSize[]={32,64,96,128,192,256};
// int arrayThreadPerIntersection[]={1,2,4,8,16,32};
// int arrayThreadShift[]={0,1,2,3,4,5};
// int arrayBlocks[]={16000};
// int arrayBlockSize[]={256};
// int arrayThreadPerIntersection[]={32};
// int arrayThreadShift[]={5};
// int arrayBlocks[]={96000};
// int arrayBlockSize[]={128,192,256};
// int arrayThreadPerIntersection[]={8,16,32};
// int arrayThreadShift[]={3,4,5};
int arrayBlocks[]={96000};
int arrayBlockSize[]={192};
int arrayThreadPerIntersection[]={8};
int arrayThreadShift[]={3};
int cutoff[]={00, 1300, 1380, 1381, 1382, 1383};

void initHostTriangleArray(triangle_t* h_triangles, vid_t nv){    
    for(vid_t sd=0; sd<(nv);sd++){
        h_triangles[sd]=0;
    }
}

int64_t sumTriangleArray(triangle_t* h_triangles, vid_t nv){    
    int64_t sum=0;
    for(vid_t sd=0; sd<(nv);sd++){
      sum+=h_triangles[sd];
    }
    return sum;
}

void testTriangleCountingConfigurations(HornetGraph& hornet, vid_t nv,degree_t ne)
{
    float minTime=10e9,time,minTimeHornet=10e9;

    int blocksToTest=sizeof(arrayBlocks)/sizeof(int);
    int blockSizeToTest=sizeof(arrayBlockSize)/sizeof(int);
    int tSPToTest=sizeof(arrayThreadPerIntersection)/sizeof(int);
    int cutoffNum = sizeof(cutoff)/sizeof(int);

    for(int b=0;b<blocksToTest; b++){
        int blocks=arrayBlocks[b];
        for(int bs=0; bs<blockSizeToTest; bs++){
            int sps=arrayBlockSize[bs];
            for(int t=0; t<tSPToTest;t++){
                int tsp=arrayThreadPerIntersection[t];
                for (int cutoff_id = 0; cutoff_id < cutoffNum ; cutoff_id ++) {
                    double running_time[10];
                    double average=0, stddev=0;
                    for (int q=0; q<10; q++) {
                        Timer<DEVICE> TM;
                        TriangleCounting tc(hornet);
                        tc.setInitParameters(blocks,sps,tsp);
                        tc.init();
                        tc.reset();

                        TM.start();
                        tc.run(cutoff[cutoff_id]);
                        TM.stop();
                        time = TM.duration();

                        triangle_t sumDevice = 0;
                        sumDevice = tc.countTriangles();
                        if(time<minTimeHornet) minTimeHornet=time; 
                        tc.release();

                        int shifter=arrayThreadShift[t];
                        int nbl=sps/tsp;

                        running_time[q] = time;
                        printf("### %d %d %d %d %d \t\t %ld \t %f\n", blocks,sps, tsp, nbl, shifter,sumDevice, time);
                        average += time;
                    }
                    average = average/10;
                    for (int q=0; q<10; q++) {
                        stddev += (running_time[q] - average) * (running_time[q] - average);
                    }
                    stddev = sqrt(stddev/10);
                    printf("cutoff = %d, average = %lf , standard deviation = %lf\n\n", cutoff[cutoff_id], average, stddev);
                }
            }
        }    
    }
    cout << nv << ", " << ne << ", "<< minTime << ", " << minTimeHornet<< endl;
}

void hostCountTriangles (const vid_t nv, const vid_t ne, const eoff_t * off,
    const vid_t * ind, int64_t* allTriangles);

int main(const int argc, char *argv[]){
 
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    int device=0;

    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
 
    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT_INFO);


    HornetInit hornet_init(graph.nV(), graph.nE(),
                                 graph.csr_out_offsets(),
                                 graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);

    hornet_graph.check_sorted_adjs();
    // std::cout << "Is sorted " <<  << std::endl;

    testTriangleCountingConfigurations(hornet_graph,graph.nV(),graph.nE());
    int64_t hostTris;
    hostCountTriangles(graph.nV(), graph.nE(),graph.csr_out_offsets(), graph.csr_out_edges(),&hostTris);
    return 0;
}






// CPU Version - assume sorted index lists. 
int hostSingleIntersection (const vid_t ai, const degree_t alen, const vid_t * a,
                            const vid_t bi, const degree_t blen, const vid_t * b){

    int32_t ka = 0, kb = 0;
     int32_t out = 0;


    if (!alen || !blen || a[alen-1] < b[0] || b[blen-1] < a[0])
    return 0;

    const vid_t *aptr=a, *aend=a+alen;
    const vid_t *bptr=b, *bend=b+blen;

    while(aptr< aend && bptr<bend){
        if(*aptr==*bptr){
            aptr++, bptr++, out++;
        }
        else if(*aptr<*bptr){
            aptr++;
        }
        else {
            bptr++;
        }
      }  
  
    return out;
}

void hostCountTriangles (const vid_t nv, const vid_t ne, const eoff_t * off,
    const vid_t * ind, int64_t* allTriangles)
{
    int32_t edge=0;
    int64_t sum=0;
    int count = 0;
    for (vid_t src = 0; src < nv; src++)
    {
        degree_t srcLen=off[src+1]-off[src];
        for(int iter=off[src]; iter<off[src+1]; iter++)
        {
            vid_t dest=ind[iter];
            degree_t destLen=off[dest+1]-off[dest];            
            if((destLen < srcLen - 1380) || destLen > srcLen + 1380) {
                count ++;
            }
            int64_t tris= hostSingleIntersection (src, srcLen, ind+off[src],
                                                    dest, destLen, ind+off[dest]);
            sum+=tris;
        }
    }    
    *allTriangles=sum;
    printf("count number %d for distance bigger than\n", count);
    printf("Sequential number of triangles %ld\n",sum);
}
