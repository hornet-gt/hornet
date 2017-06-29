/*#include "update.hpp"
#include "cuStinger.hpp"
#include "operators.cuh"
#include "static_k_truss/k_truss.cuh"*/
#include "Static/KTruss/KTruss.cuh"
#include "Support/Device/CudaUtil.cuh"

namespace custinger_alg {

void kTrussOneIteration(cuStinger& custinger,
                        const triangle_t*  __restrict__ outPutTriangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        KTrussData* __restrict__ devData);

KTruss::KTruss(cuStinger& custinger, custinger::BatchUpdate& batch_update) :
                                        StaticAlgorithm(custinger),
                                        hostKTrussData(custinger),
                                        batch_update(batch_update) {}

KTruss::~KTruss() {
    release();
}

void KTruss::setInitParameters(vid_t nv, eoff_t ne, int tsp, int nbl,
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

    //hostKTrussData.activeQueue.init(hostKTrussData.nv + 1);

    syncDeviceWithHost();
    reset();
}

void KTruss::copyOffsetArrayHost(vid_t* hostOffsetArray) {
    //copyArrayHostToDevice(hostOffsetArray, hostKTrussData.offsetArray,
    //                      hostKTrussData.nv+1, sizeof(length_t));
    cuMemcpyToDevice(hostOffsetArray, hostKTrussData.nv + 1,
                     hostKTrussData.offsetArray);
}

void KTruss::copyOffsetArrayDevice(vid_t* deviceOffsetArray){
    //copyArrayDeviceToDevice(deviceOffsetArray, hostKTrussData.offsetArray,
    //                        hostKTrussData.nv + 1, sizeof(vid_t));
    cuMemcpyToDevice(deviceOffsetArray, hostKTrussData.nv + 1,
                     hostKTrussData.offsetArray);
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
    hostKTrussData.isActive = nullptr;
    hostKTrussData.offsetArray = nullptr;
    hostKTrussData.trianglePerEdge  = nullptr;
    hostKTrussData.trianglePerVertex = nullptr;
}

//==============================================================================

void KTruss::run() {
    hostKTrussData.maxK = 3;
    syncDeviceWithHost();

    while (true) {
        bool needStop = false;
        bool     more = findTrussOfK(needStop);
        if (more == false && needStop) {
            hostKTrussData.maxK--;
            syncDeviceWithHost();
            break;
        }
        hostKTrussData.maxK++;
        syncDeviceWithHost();
    }
    // cout << "Found the maximal KTruss at : " << hostKTrussData.maxK << endl;
    std::cout << "The number of full triangle counting iterations is  : "
              << hostKTrussData.fullTriangleIterations << std::endl;
}

void KTruss::runForK(int maxK) {
    hostKTrussData.maxK = maxK;
    syncDeviceWithHost();

    bool exitOnFirstIteration;
    findTrussOfK(exitOnFirstIteration);
}


bool KTruss::findTrussOfK(bool& stop) {
    forAllVertices<ktruss_operators::init>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<ktruss_operators::init>(custinger,deviceKTrussData);

    resetEdgeArray();
    resetVertexArray();

    hostKTrussData.counter = 0;
    hostKTrussData.activeVertices = custinger.nV();
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

        forAllVertices<ktruss_operators::findUnderK>(custinger,deviceKTrussData);
        //allVinG_TraverseVertices<ktruss_operators::findUnderK>(custinger,deviceKTrussData);
        syncHostWithDevice();
        // cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;
        sumDeletedEdges += hostKTrussData.counter;
        if (hostKTrussData.counter == hostKTrussData.ne_remaining) {
            stop = true;
            return false;
        }
        if (hostKTrussData.counter != 0) {
            //BatchUpdateData* bud = new BatchUpdateData(hostKTrussData.counter,
            //                                           true, hostKTrussData.nv);
            //BatchUpdateData bud(hostKTrussData.counter, true, hostKTrussData.nv);

            auto src_array = new vid_t[hostKTrussData.counter];
            auto dst_array = new vid_t[hostKTrussData.counter];

            cuMemcpyToHost(hostKTrussData.src,  hostKTrussData.counter,
                           src_array);
            cuMemcpyToHost(hostKTrussData.dst, hostKTrussData.counter,
                           dst_array);
            /*copyArrayDeviceToHost(hostKTrussData.src, bud->getSrc(),
                                  hostKTrussData.counter, sizeof(int));
            copyArrayDeviceToHost(hostKTrussData.dst, bud->getDst(),
                                  hostKTrussData.counter, sizeof(int));*/
            custinger::BatchInit batch_init(src_array, dst_array,
                                            hostKTrussData.counter);
            //custinger::BatchUpdate batch_update(batch_init);
            batch_update.insert(batch_init);
            custinger.edgeDeletionsSorted(batch_update);    ///???

            //batch_update->sortDeviceBUD(hostKTrussData.sps);
            //custinger.edgeDeletionsSorted(*batch_update);

            delete[] src_array;
            delete[] dst_array;
        }
        else
            return false;

        hostKTrussData.ne_remaining  -= hostKTrussData.counter;
        hostKTrussData.activeVertices = 0;

        syncDeviceWithHost();

        forAllVertices<ktruss_operators::countActive>
            (custinger, deviceKTrussData);

        //allVinG_TraverseVertices<ktruss_operators::countActive>
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

void KTruss::runDynamic(){
    hostKTrussData.maxK = 3;
    syncDeviceWithHost();
    CHECK_CUDA_ERROR
    forAllVertices<ktruss_operators::init>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<ktruss_operators::init>(custinger, deviceKTrussData);
    CHECK_CUDA_ERROR

    resetEdgeArray();
    resetVertexArray();
    syncDeviceWithHost();

    kTrussOneIteration(custinger, hostKTrussData.trianglePerVertex, 4,
                       hostKTrussData.sps / 4, 2, hostKTrussData.blocks,
                       hostKTrussData.sps, deviceKTrussData);
    CHECK_CUDA_ERROR
    syncHostWithDevice();

    forAllVertices<ktruss_operators::resetWeights>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<ktruss_operators::resetWeights>
    //    (custinger, deviceKTrussData);
    CHECK_CUDA_ERROR
    while (true) {
        // if(hostKTrussData.maxK >=5)
        // break;
        // cout << "New iteration" << endl;
        bool needStop = false;
        bool     more = findTrussOfKDynamic(needStop);
        CHECK_CUDA_ERROR
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

bool KTruss::findTrussOfKDynamic(bool& stop) {
    hostKTrussData.counter = 0;
    syncDeviceWithHost();

    //hostKTrussData.activeQueue.clear();
    syncDeviceWithHost();

    forAllVertices<ktruss_operators::queueActive>(custinger, deviceKTrussData);
    forAllVertices<ktruss_operators::countActive>(custinger, deviceKTrussData);
    CHECK_CUDA_ERROR
    //allVinG_TraverseVertices<ktruss_operators::queueActive>
    //    (custinger, deviceKTrussData);
    //allVinG_TraverseVertices<ktruss_operators::countActive>
    //    (custinger, deviceKTrussData);
    syncHostWithDevice();

    //int activeThisIteration = hostKTrussData.activeQueue.getQueueEnd();
    int activeThisIteration = hostKTrussData.activeQueue.size();
    stop       = true;
    //bool first = true;
    while (hostKTrussData.activeVertices > 0) {
        //allVinA_TraverseVertices<ktruss_operators::findUnderKDynamic>
        //    (custinger, deviceKTrussData, hostKTrussData.activeQueue.getQueue(),
        //     activeThisIteration);
        CHECK_CUDA_ERROR
        //forAllVertices<ktruss_operators::findUnderKDynamic>
        //    (custinger, hostKTrussData.activeQueue, deviceKTrussData);    //???
        forAllVertices<ktruss_operators::findUnderKDynamic>
            (custinger, deviceKTrussData);    //???
        CHECK_CUDA_ERROR

        syncHostWithDevice();
        // cout << "Current number of deleted edges is " << hostKTrussData.counter << endl;

        if (hostKTrussData.counter == hostKTrussData.ne_remaining) {
            stop = true;
            return false;
        }
        if (hostKTrussData.counter != 0) {
            //BatchUpdateData bud(hostKTrussData.counter,true, hostKTrussData.nv);

            auto src_array = new vid_t[hostKTrussData.counter];
            auto dst_array = new vid_t[hostKTrussData.counter];

            cuMemcpyToHost(hostKTrussData.src, hostKTrussData.counter,
                           src_array);
            cuMemcpyToHost(hostKTrussData.dst, hostKTrussData.counter,
                           dst_array);
            //copyArrayDeviceToHost(hostKTrussData.src,bud->getSrc(),
            //                      hostKTrussData.counter, sizeof(int));
            //copyArrayDeviceToHost(hostKTrussData.dst, bud->getDst(),
            //                      hostKTrussData.counter, sizeof(int));
            custinger::BatchInit batch_init(src_array, dst_array,
                                            hostKTrussData.counter);
            //custinger::BatchUpdate batch_update(batch_init);
            batch_update.insert(batch_init);

            custinger.edgeDeletionsSorted(batch_update);
            //custinger::BatchUpdate batch_update(*bud);
            //batch_update->sortDeviceBUD(hostKTrussData.sps);
            //custinger.edgeDeletionsSorted(*batch_update);

            callDeviceDifferenceTriangles(custinger, batch_update,
                                          hostKTrussData.trianglePerVertex,
                                          hostKTrussData.tsp,
                                          hostKTrussData.nbl,
                                          hostKTrussData.shifter,
                                          hostKTrussData.blocks,
                                          hostKTrussData.sps, true);
            delete[] src_array;
            delete[] dst_array;
        }
        else
            return false;
CHECK_CUDA_ERROR
        hostKTrussData.ne_remaining  -= hostKTrussData.counter;
        hostKTrussData.activeVertices = 0;
        hostKTrussData.counter        = 0;
        syncDeviceWithHost();

        //allVinA_TraverseVertices<ktruss_operators::countActive>
        //    (custinger, deviceKTrussData, hostKTrussData.activeQueue.getQueue(),
        //     activeThisIteration);
        //forAllVertices<ktruss_operators::countActive>
        //    (custinger, hostKTrussData.activeQueue, deviceKTrussData);  //???
        forAllVertices<ktruss_operators::countActive>
            (custinger, deviceKTrussData);  //???
CHECK_CUDA_ERROR
        syncHostWithDevice();
        stop = false;
    }
    return true;
}

void KTruss::runForKDynamic(int maxK) {
    hostKTrussData.maxK = maxK;
    syncDeviceWithHost();

    forAllVertices<ktruss_operators::init>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<ktruss_operators::init>(custing,deviceKTrussData);

    resetEdgeArray();
    resetVertexArray();
    syncDeviceWithHost();

    kTrussOneIteration(custinger, hostKTrussData.trianglePerVertex, 4,
                       hostKTrussData.sps / 4, 2, hostKTrussData.blocks,
                       hostKTrussData.sps, deviceKTrussData);

    syncHostWithDevice();

    forAllVertices<ktruss_operators::resetWeights>(custinger, deviceKTrussData);
    //allVinG_TraverseVertices<ktruss_operators::resetWeights>(custing,deviceKTrussData);

    bool needStop = false;
    bool     more = findTrussOfKDynamic(needStop);
    // cout << "Found the maximal KTruss at : " << hostKTrussData.maxK << endl;
}

}// custinger_alg namespace
