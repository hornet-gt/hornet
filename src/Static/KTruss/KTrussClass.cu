#include <stdio.h>
#include <inttypes.h>

#include "timer.h"
#include "update.hpp"
#include "cuStinger.hpp"
#include "operators.cuh"
#include "static_k_truss/k_truss.cuh"

namespace custinger_alg {

void kTrussOneIteration(cuStinger& custinger,
                        const triangle_t*  __restrict__ outPutTriangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        kTrussData* __restrict__ devData);

void callDeviceDifferenceTriangles(
                                cuStinger& custinger,
                                BatchUpdate& batch_update,
                                const triangle_t* __restrict__ outPutTriangles,
                                int threads_per_intersection,
                                int num_intersec_perblock,
                                int shifter,
                                int thread_blocks,
                                int blockdim,
                                bool deletion);

void KTruss::setInitParameters(vid_t nv, vid_t ne, int tsp, int nbl,
                               int shifter, int blocks, int sps) {
    hostKTrussData.nv      = nv;
    hostKTrussData.ne      = ne;
    hostKTrussData.tsp     = tsp;
    hostKTrussData.nbl     = nbl;
    hostKTrussData.shifter = shifter;
    hostKTrussData.blocks  = blocks;
    hostKTrussData.sps     = sps;
}

void KTruss::init(){
    cuMalloc(hostKTrussData.isActive, hostKTrussData.nv);
    cuMalloc(hostKTrussData.offsetArray, hostKTrussData.nv + 1);
    cuMalloc(hostKTrussData.trianglePerVertex, hostKTrussData.nv);
    cuMalloc(hostKTrussData.trianglePerEdge, hostKTrussData.ne);
    cuMalloc(hostKTrussData.src, hostKTrussData.ne);
    cuMalloc(hostKTrussData.dst, hostKTrussData.ne);
    //cuMalloc(deviceKTrussData, 1);
    deviceKTrussData = register_data(hostKTrussData);

    hostKTrussData.activeQueue.init(hostKTrussData.nv + 1);

    syncDeviceWithHost();
    reset();
}

void KTruss::copyOffsetArrayHost(vid_t* hostOffsetArray) {
    cuMemcpyToDevice(hostOffsetArray, hostKTrussData.offsetArray,
                     hostKTrussData.nv + 1);
}

void KTruss::copyOffsetArrayDevice(vid_t* deviceOffsetArray){
    copyArrayDeviceToDevice(deviceOffsetArray, hostKTrussData.offsetArray,
                            hostKTrussData.nv + 1, sizeof(vid_t));
}

vid_t KTruss::getMaxK() {
    return hostKTrussData.maxK;
}

//==============================================================================

void KTruss::reset() {
    hostKTrussData.counter                = 0;
    hostKTrussData.ne_remaining           = hostKTrussData.ne;
    hostKTrussData.fullTriangleIterations = 0;

    resetEdgeArray();
    resetVertexArray();
    syncDeviceWithHost();
}

void KTruss::resetVertexArray() {
    cuMemset0x00(hostKTrussData.trianglePerVertex, hostKTrussData.nv);
}

void KTruss::resetEdgeArray() {
    cuMemset0x00(hostKTrussData.trianglePerEdge, hostKTrussData.ne);
}

void KTruss::release() {
    cuFree(hostKTrussData.isActive, hostKTrussData.offsetArray,
           hostKTrussData.trianglePerEdge, hostKTrussData.trianglePerVertex);
}

//==============================================================================

void KTruss::run(cuStinger& custinger){
    hostKTrussData.maxK = 3;
    syncDeviceWithHost();

    while (true) {
        bool needStop = false;
        bool     more = findTrussOfK(custinger, needStop);
        if (more == false && needStop) {
            hostKTrussData.maxK--;
            syncDeviceWithHost();
            break;
        }
        hostKTrussData.maxK++;
        syncDeviceWithHost();
    }
    // cout << "Found the maximal KTruss at : " << hostKTrussData.maxK << endl;
    cout << "The number of full triangle counting iterations is  : "
         << hostKTrussData.fullTriangleIterations << endl;
}

void KTruss::runForK(cuStinger& custinger, int maxK) {
    hostKTrussData.maxK = maxK;
    syncDeviceWithHost();

    bool exitOnFirstIteration;
    findTrussOfK(custinger,exitOnFirstIteration);
}


bool KTruss::findTrussOfK(cuStinger& custinger, bool& stop) {
    forAllVertices<kTrussOperators::init>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<kTrussOperators::init>(custinger,deviceKTrussData);

    resetEdgeArray();
    resetVertexArray();

    hostKTrussData.counter = 0;
    hostKTrussData.activeVertices = custinger.nv;
    syncDeviceWithHost();

    int sumDeletedEdges = 0;
    stop = true;

    while (hostKTrussData.activeVertices > 0) {
        hostKTrussData.fullTriangleIterations++;
        syncDeviceWithHost();

        kTrussOneIteration(custinger, hostKTrussData.trianglePerVertex,
                           hostKTrussData.tsp, hostKTrussData.nbl,
                           hostKTrussData.shifter,
                           hostKTrussData.blocks, hostKTrussData.sps,
                           deviceKTrussData);

        forAllVertices<kTrussOperators::findUnderK>(custinger,deviceKTrussData);
        //allVinG_TraverseVertices<kTrussOperators::findUnderK>(custinger,deviceKTrussData);
        syncHostWithDevice();
        // cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;
        sumDeletedEdges += hostKTrussData.counter;
        if (hostKTrussData.counter == hostKTrussData.ne_remaining) {
            stop = true;
            return false;
        }
        if (hostKTrussData.counter != 0) {
            BatchUpdateData* bud = new BatchUpdateData(hostKTrussData.counter,
                                                       true, hostKTrussData.nv);

            cuMemcpyToHost(hostKTrussData.src, bud->getSrc(),
                           hostKTrussData.counter);
            cuMemcpyToHost(hostKTrussData.dst, bud->getDst(),
                           hostKTrussData.counter);
            /*copyArrayDeviceToHost(hostKTrussData.src, bud->getSrc(),
                                  hostKTrussData.counter, sizeof(int));
            copyArrayDeviceToHost(hostKTrussData.dst, bud->getDst(),
                                  hostKTrussData.counter, sizeof(int));*/
            BatchUpdate* batch_update = new BatchUpdate(*bud);

            batch_update->sortDeviceBUD(hostKTrussData.sps);
            custinger.edgeDeletionsSorted(*batch_update);
            delete batch_update;
            delete bud;
        }
        else
            return false;

        hostKTrussData.ne_remaining  -= hostKTrussData.counter;
        hostKTrussData.activeVertices = 0;

        syncDeviceWithHost();

        forAllVertices<kTrussOperators::countActive>
            (custinger, deviceKTrussData);

        //allVinG_TraverseVertices<kTrussOperators::countActive>
        //    (custinger, deviceKTrussData);
        syncHostWithDevice();
        resetEdgeArray();
        resetVertexArray();

        hostKTrussData.counter = 0;
        syncDeviceWithHost();
        stop = false;
    }
    return true;
}

void KTruss::runDynamic(cuStinger& custinger){
    hostKTrussData.maxK = 3;
    syncDeviceWithHost();

    forAllVertices<kTrussOperators::init>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<kTrussOperators::init>(custinger, deviceKTrussData);

    resetEdgeArray();
    resetVertexArray();
    syncDeviceWithHost();

    kTrussOneIteration(custinger, hostKTrussData.trianglePerVertex, 4,
                       hostKTrussData.sps / 4, 2, hostKTrussData.blocks,
                       hostKTrussData.sps, deviceKTrussData);

    syncHostWithDevice();

    forAllVertices<kTrussOperators::resetWeights>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<kTrussOperators::resetWeights>
    //    (custinger, deviceKTrussData);

    while (true) {
        // if(hostKTrussData.maxK >=5)
        // break;
        // cout << "New iteration" << endl;
        bool needStop = false;
        bool     more = findTrussOfKDynamic(custinger, needStop);
        if (more == false && needStop) {
            hostKTrussData.maxK--;
            syncDeviceWithHost();
            break;
        }
        hostKTrussData.maxK++;
        syncDeviceWithHost();
    }
    // cout << "Found the maximal KTruss at : " << hostKTrussData.maxK << endl;
}

bool KTruss::findTrussOfKDynamic(cuStinger& custinger, bool& stop) {
    hostKTrussData.counter = 0;
    syncDeviceWithHost();

    hostKTrussData.activeQueue.clear();
    syncDeviceWithHost();

    forAllVertices<kTrussOperators::queueActive>(custinger, deviceKTrussData);
    forAllVertices<kTrussOperators::countActive>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<kTrussOperators::queueActive>
    //    (custinger, deviceKTrussData);
    //allVinG_TraverseVertices<kTrussOperators::countActive>
    //    (custinger, deviceKTrussData);
    syncHostWithDevice();

    int activeThisIteration = hostKTrussData.activeQueue.getQueueEnd();//???
    stop       = true;
    bool first = true;
    while (hostKTrussData.activeVertices > 0) {
        allVinA_TraverseVertices<kTrussOperators::findUnderKDynamic>
            (custinger, deviceKTrussData, hostKTrussData.activeQueue.getQueue(),
             activeThisIteration);
        // allVinA_TraverseEdges_Weights<kTrussOperators::findUnderKDynamicWeights>(custinger, deviceKTrussData, hostKTrussData.activeQueue.getQueue(), activeThisIteration);
        // allVinG_TraverseVertices<kTrussOperators::findUnderKDynamic>(custinger,deviceKTrussData);

        // allVinA_TraverseEdges_LB_Weight<kTrussOperators::findUnderKDynamicWeights>(custinger,deviceKTrussData, *cusLB,hostKTrussData.activeQueue, first);
        // first=false;
        syncHostWithDevice();
        // cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;

        if (hostKTrussData.counter == hostKTrussData.ne_remaining) {
            stop = true;
            return false;
        }
        if (hostKTrussData.counter != 0) {
            BatchUpdateData bud(hostKTrussData.counter,true, hostKTrussData.nv);

            cuMemcpyToHost(hostKTrussData.src, bud->getSrc(),
                           hostKTrussData.counter);
            cuMemcpyToHost(hostKTrussData.dst, bud->getDst(),
                           hostKTrussData.counter);
            //copyArrayDeviceToHost(hostKTrussData.src,bud->getSrc(),
            //                      hostKTrussData.counter, sizeof(int));
            //copyArrayDeviceToHost(hostKTrussData.dst, bud->getDst(),
            //                      hostKTrussData.counter, sizeof(int));

            BatchUpdate batch_update(*bud);
            batch_update->sortDeviceBUD(hostKTrussData.sps);

            custinger.edgeDeletionsSorted(*batch_update);

            callDeviceDifferenceTriangles(custinger, *batch_update,
                                          hostKTrussData.trianglePerVertex,
                                          hostKTrussData.tsp,
                                          hostKTrussData.nbl,
                                          hostKTrussData.shifter,
                                          hostKTrussData.blocks,
                                          hostKTrussData.sps, true);
        }
        else
            return false;

        hostKTrussData.ne_remaining  -= hostKTrussData.counter;
        hostKTrussData.activeVertices = 0;
        hostKTrussData.counter        = 0;
        syncDeviceWithHost();

        allVinA_TraverseVertices<kTrussOperators::countActive>
            (custinger, deviceKTrussData, hostKTrussData.activeQueue.getQueue(),
             activeThisIteration);

        syncHostWithDevice();
        stop = false;
    }
    return true;
}

void KTruss::runForKDynamic(cuStinger& custinger, int maxK) {
    hostKTrussData.maxK = maxK;
    syncDeviceWithHost();

    forAllVertices<kTrussOperators::init>(custing, deviceKTrussData);
    //allVinG_TraverseVertices<kTrussOperators::init>(custing,deviceKTrussData);

    resetEdgeArray();
    resetVertexArray();
    syncDeviceWithHost();

    kTrussOneIteration(custing, hostKTrussData.trianglePerVertex, 4,
                       hostKTrussData.sps / 4, 2, hostKTrussData.blocks,
                       hostKTrussData.sps, deviceKTrussData);

    syncHostWithDevice();

    forAllVertices<kTrussOperators::resetWeights>(custing,deviceKTrussData);
    //allVinG_TraverseVertices<kTrussOperators::resetWeights>(custing,deviceKTrussData);

    bool needStop = false;
    bool     more = findTrussOfKDynamic(custing, needStop);
    // cout << "Found the maximal KTruss at : " << hostKTrussData.maxK << endl;
}

}// custinger_alg namespace
