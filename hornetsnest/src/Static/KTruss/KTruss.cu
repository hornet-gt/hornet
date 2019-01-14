#include "Static/KTruss/KTruss.cuh"
#include "KTrussOperators.cuh"

namespace hornets_nest {

void kTrussOneIteration(HornetGraph& hornet,
                        const triangle_t*  __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data);

//==============================================================================

KTruss::KTruss(HornetGraph& hornet) : StaticAlgorithm(hornet) {
    hd_data().active_queue.initialize(hornet);
}

KTruss::~KTruss() {
    release();
}

void KTruss::setInitParameters(int tsp, int nbl, int shifter,
                               int blocks, int sps) {
    hd_data().tsp     = tsp;
    hd_data().nbl     = nbl;
    hd_data().shifter = shifter;
    hd_data().blocks  = blocks;
    hd_data().sps     = sps;
}

void KTruss::init(){
    gpu::allocate(hd_data().is_active,            hornet.nV());
    gpu::allocate(hd_data().offset_array,         hornet.nV() + 1);
    gpu::allocate(hd_data().triangles_per_vertex, hornet.nV());
    gpu::allocate(hd_data().triangles_per_edge,   hornet.nE());
    gpu::allocate(hd_data().src,                  hornet.nE());
    gpu::allocate(hd_data().dst,                  hornet.nE());
    reset();
}

void KTruss::copyOffsetArrayHost(vid_t* host_offset_array) {
    host::copyToDevice(host_offset_array, hornet.nV() + 1,
                       hd_data().offset_array);
}

void KTruss::copyOffsetArrayDevice(vid_t* device_offset_array){
    host::copyToDevice(device_offset_array, hornet.nV() + 1,
                       hd_data().offset_array);
}

vid_t KTruss::getMaxK() {
    return hd_data().max_K;
}

//==============================================================================

void KTruss::reset() {
    hd_data().counter                  = 0;
    hd_data().num_edges_remaining      = hornet.nE();
    hd_data().full_triangle_iterations = 0;

    resetEdgeArray();
    resetVertexArray();
}

void KTruss::resetVertexArray() {
    gpu::memsetZero(hd_data().triangles_per_vertex, hornet.nV());
}

void KTruss::resetEdgeArray() {
    gpu::memsetZero(hd_data().triangles_per_edge, hornet.nE());
}

void KTruss::release() {
    gpu::free(hd_data().is_active);
    gpu::free(hd_data().offset_array);
    gpu::free(hd_data().triangles_per_edge);
    gpu::free(hd_data().triangles_per_vertex);
    hd_data().is_active            = nullptr;
    hd_data().offset_array         = nullptr;
    hd_data().triangles_per_edge   = nullptr;
    hd_data().triangles_per_vertex = nullptr;
}

//==============================================================================

void KTruss::run() {
    hd_data().max_K = 3;
    int  iterations = 0;

    while (true) {
        // if(hd_data().max_K >= 5)
        //     break;
        //std::cout << hd_data().num_edges_remaining << std::endl;
        bool need_stop = false;
        bool      more = findTrussOfK(need_stop);
        //if (more == false && need_stop) {
        if (hd_data().num_edges_remaining <= 0) {
            hd_data().max_K--;
            break;
        }
        hd_data().max_K++;
        iterations++;
    }
    //std::cout << "iterations " << iterations << std::endl;
    // cout << "Found the maximal KTruss at : " << hd_data().max_K << endl;
    //std::cout << "The number of full triangle counting iterations is  : "
    //          << hd_data().full_triangle_iterations << std::endl;
}

void KTruss::runForK(int max_K) {
    hd_data().max_K = max_K;

    bool exit_on_first_iteration;
    findTrussOfK(exit_on_first_iteration);
}

bool KTruss::findTrussOfK(bool& stop) {
    forAllVertices(hornet, Init { hd_data });
    resetEdgeArray();
    resetVertexArray();

    hd_data().counter = 0;
    hd_data().active_vertices = hornet.nV();

    int sum_deleted_edges = 0;
    stop = true;

    while (hd_data().active_vertices > 0) {
        hd_data().full_triangle_iterations++;

        kTrussOneIteration(hornet, hd_data().triangles_per_vertex,
                           hd_data().tsp, hd_data().nbl,
                           hd_data().shifter,
                           hd_data().blocks, hd_data().sps,
                           hd_data);

        forAllVertices(hornet, FindUnderK { hd_data });
        hd_data.sync();
        //std::cout << "Current number of deleted edges is " <<
        //hd_data().counter << std::endl;

        sum_deleted_edges += hd_data().counter;
        /*if (hd_data().counter == hd_data().num_edges_remaining) {
            stop = true;
            return false;
        }*/
        if (hd_data().counter != 0) {
            //directly on the device
            BatchUpdate batch_update(hd_data().src, hd_data().dst,
                                     hd_data().counter, gpu::BatchType::DEVICE);

            hornet.deleteEdgeBatch(batch_update);
        }
        else
            return false;

        hd_data().num_edges_remaining -= hd_data().counter;
        hd_data().active_vertices = 0;

        forAllVertices(hornet, CountActive { hd_data });

        hd_data.sync();
        resetEdgeArray();
        resetVertexArray();

        hd_data().counter = 0;
        stop = false;
    }
    return true;
}

//==============================================================================
//==============================================================================
//==============================================================================

void KTruss::runDynamic(){
    hd_data().max_K = 3;
    forAllVertices(hornet, Init { hd_data });

    resetEdgeArray();
    resetVertexArray();

    kTrussOneIteration(hornet, hd_data().triangles_per_vertex,
                           hd_data().tsp, hd_data().nbl,
                           hd_data().shifter,
                           hd_data().blocks, hd_data().sps,
                           hd_data);   //sub
    hd_data.sync();
    forAllVertices(hornet, ResetWeights { hd_data });

    int iterations = 0;
    while (true) {
        //if(hd_data().max_K >= 5)
        //    break;
        //std::cout << "New iteration" << std::endl;
        bool need_stop = false;
        bool     more = findTrussOfKDynamic(need_stop);
        CHECK_CUDA_ERROR
        //std::cout << hd_data().num_edges_remaining << std::endl;
        //if (more == false && need_stop) {
        if (hd_data().num_edges_remaining <= 0) {
            hd_data().max_K--;
            break;
        }
        hd_data().max_K++;
        iterations++;
    }
    //std::cout << "iterations " << iterations << std::endl;
}

bool KTruss::findTrussOfKDynamic(bool& stop) {
    hd_data().counter = 0;
    hd_data().active_queue.clear();  //queue

    forAllVertices(hornet, QueueActive { hd_data }); //queue
    forAllVertices(hornet, CountActive { hd_data });
    hd_data.sync();
    hd_data().active_queue.swap();//queue

    stop = true;
    while (hd_data().active_vertices > 0) {
        forAllVertices(hornet, hd_data().active_queue,
                       FindUnderKDynamic { hd_data });    //queue
        hd_data.sync();
        //std::cout << "Current number of deleted edges is "
        //<< hd_data().counter << std::endl;

        /*if (hd_data().counter == hd_data().num_edges_remaining) {
            stop = true;
            return false;
        }*/
        if (hd_data().counter != 0) {
            //directly on the device
            //auto src_array = new vid_t[hd_data().counter];
            //auto dst_array = new vid_t[hd_data().counter];
            //cuMemcpyToHost(hd_data().src, hd_data().counter,
            //               src_array);
            //cuMemcpyToHost(hd_data().dst, hd_data().counter,
            //               dst_array);
            //hornet::BatchInit batch_init(src_array, dst_array,
            //                                hd_data().counter);
            BatchUpdate batch_update(hd_data().src, hd_data().dst,
                                     hd_data().counter, gpu::BatchType::DEVICE);
            //batch_update.sendToDevice(batch_init);//directly on the device
            hornet.deleteEdgeBatch(batch_update);

            callDeviceDifferenceTriangles(hornet, batch_update,
                                          hd_data().triangles_per_vertex,
                                          hd_data().tsp,
                                          hd_data().nbl,
                                          hd_data().shifter,
                                          hd_data().blocks,
                                          hd_data().sps, true);
        }
        else
            return false;

        hd_data().num_edges_remaining -= hd_data().counter;
        hd_data().active_vertices = 0;
        hd_data().counter         = 0;

        //allVinA_TraverseVertices<ktruss_operators::countActive>
        //    (hornet, deviceKTrussData, hd_data().active_queue.getQueue(),
        //     activeThisIteration);

        forAllVertices(hornet, hd_data().active_queue, CountActive { hd_data });

        hd_data.sync();
        stop = false;
    }
    return true;
}

void KTruss::runForKDynamic(int max_K) {
    hd_data().max_K = max_K;

    forAllVertices(hornet, Init { hd_data });

    resetEdgeArray();
    resetVertexArray();

    kTrussOneIteration(hornet, hd_data().triangles_per_vertex, 4,
                       hd_data().sps / 4, 2, hd_data().blocks,
                       hd_data().sps, hd_data);
    hd_data.sync();

    forAllVertices(hornet, ResetWeights { hd_data });

    bool need_stop = false;
    bool      more = findTrussOfKDynamic(need_stop);
}

} // hornet_alg namespace
