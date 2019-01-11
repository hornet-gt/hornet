
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include "Static/TriangleCounting/triangle.cuh"
#include <StandardAPI.hpp>
#include <Device/Util/Timer.cuh>

#include <Graph/GraphStd.hpp>

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
int arrayThreadPerIntersection[]={16};
int arrayThreadShift[]={3};

int cutoff[]={-1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
1400, 1500, 1600, 1700, 1800, 1900, 2000,
2100, 2200, 2300, 2400,
2500, 2600, 2700
};

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

void testTriangleCountingConfigurations(HornetGraph& hornet, vid_t nv,degree_t ne, int *histogram)
{
    float minTime=10e9,time,minTimeHornet=10e9;

    int blocksToTest=sizeof(arrayBlocks)/sizeof(int);
    int blockSizeToTest=sizeof(arrayBlockSize)/sizeof(int);
    int tSPToTest=sizeof(arrayThreadPerIntersection)/sizeof(int);

    for(int b=0;b<blocksToTest; b++){
        int blocks=arrayBlocks[b];
        for(int bs=0; bs<blockSizeToTest; bs++){
            int sps=arrayBlockSize[bs];
            for(int t=0; t<tSPToTest;t++){
                int tsp=arrayThreadPerIntersection[t];
                double prev_average = 0;
                for (auto cutoff_id : cutoff) {
                    double running_time[10];
                    double average=0, stddev=0;
                    for (int q=0; q<10; q++) {
                        Timer<DEVICE> TM;
                        TriangleCounting tc(hornet);
                        tc.setInitParameters(blocks,sps,tsp);
                        tc.init();
                        tc.reset();

                        TM.start();
                        tc.run(cutoff_id);
                        TM.stop();
                        time = TM.duration();

                        triangle_t sumDevice = 0;
                        sumDevice = tc.countTriangles();
                        if(time<minTimeHornet) minTimeHornet=time; 
                        tc.release();

                        int shifter=arrayThreadShift[t];
                        int nbl=sps/tsp;

                        running_time[q] = time;
                        printf("### %d %d %d %d %d \t\t %u \t %f\n", blocks,sps, tsp, nbl, shifter, sumDevice, time);
                        average += time;
                    }
                    average = average/10;
                    for (int q=0; q<10; q++) {
                        stddev += (running_time[q] - average) * (running_time[q] - average);
                    }
                    stddev = sqrt(stddev/10);
                    auto diff = cutoff_id/100;
                    if (diff > 2600) diff = 2600;
                    double rate = histogram[diff]/(average-prev_average);
                    prev_average = average;
                    printf("cutoff = %d, rate = %lf, average = %lf , standard deviation = %lf\n", cutoff_id, rate, average, stddev);
                }
            }
        }    
    }
    cout << nv << ", " << ne << ", "<< minTime << ", " << minTimeHornet<< endl;
}

// CPU Version - assume sorted index lists. 
int hostSingleIntersection (const vid_t ai, const degree_t alen, const vid_t * a,
                            const vid_t bi, const degree_t blen, const vid_t * b){

    //int32_t ka = 0, kb = 0;
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

int* hostCountTriangles (const vid_t nv, const vid_t ne, const eoff_t * off,
    const vid_t * ind, int64_t* allTriangles)
{
    //int32_t edge=0;
    int64_t sum=0;
    int count = 0;
    int *histogram = new int[27]();
    degree_t maxd = 0;
    for (vid_t src = 0; src < nv; src++)
    {
        degree_t srcLen=off[src+1]-off[src];
        for(int iter=off[src]; iter<off[src+1]; iter++)
        {
            vid_t dest=ind[iter];
            degree_t destLen=off[dest+1]-off[dest];
            if (destLen+srcLen > maxd) maxd = destLen+srcLen;
            size_t diff = abs(destLen+srcLen);
            if (diff > 2600) diff = 2600;
            histogram[diff/100] ++;
            if((destLen < srcLen - 1380) || destLen > srcLen + 1380) {
                count ++;
            }
            //int64_t tris= hostSingleIntersection (src, srcLen, ind+off[src],
            //                                        dest, destLen, ind+off[dest]);
            //sum+=tris;
        }
    }    
    printf("max: %d\n", maxd);
    for(int i=0; i<27; i++) 
        printf("histogram %d: %d\n", i, histogram[i]);

    *allTriangles=sum;
    printf("count number %d for distance bigger than\n", count);
    printf("Sequential number of triangles %ld\n",sum);
    return histogram;
}

int exec(const int argc, char *argv[]){
 
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

    std::cout << "Initializing GPU graph" << std::endl;
    HornetGraph hornet_graph(hornet_init);
    std::cout << "Checking sortd adj" << std::endl;

    hornet_graph.check_sorted_adjs();
    // std::cout << "Is sorted " <<  << std::endl;

    int64_t hostTris;
    std::cout << "Starting host triangle counting" << std::endl;
    int *histogram = hostCountTriangles(graph.nV(), graph.nE(),graph.csr_out_offsets(), graph.csr_out_edges(),&hostTris);
    testTriangleCountingConfigurations(hornet_graph,graph.nV(),graph.nE(),histogram);
    delete histogram;
    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
#if defined(RMM_WRAPPER)
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

