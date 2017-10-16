#include "Static/KTruss/KTruss.cuh"
#include "Device/Timer.cuh"
#include <GraphIO/GraphStd.hpp>

using namespace timer;
using namespace hornets_nest;

void runKtruss(const HornetInit& hornet_init, int alg, int max_K,
               const std::string& graph_name);

int main(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT_INFO);

    HornetInit hornet_init(graph.nV(), graph.nE(),
                           graph.out_offsets_ptr(),
                           graph.out_edges_ptr(), true);

    int alg = 3, max_K = 3;
    if (argc >= 3)
        alg = std::stoi(argv[2]);
    if (argc >= 4)
        max_K = std::stoi(argv[3]);

    auto weights = new int[graph.nE()]();
    hornet_init.insertEdgeData(weights);
    runKtruss(hornet_init, alg, max_K, graph.name());
    delete[] weights;
}

//==============================================================================

const int                arrayBlocks[] = { 16000 };
const int             arrayBlockSize[] = { 32 };
const int arrayThreadPerIntersection[] = { 1 };
const int           arrayThreadShift[] = { 0 };

void runKtruss(const HornetInit& hornet_init, int alg, int max_K,
               const std::string& graph_name) {
    using namespace gpu::batch_property;
    using namespace hornets_nest;

    int nV = hornet_init.nV();
    int nE = hornet_init.nE();

    triangle_t* d_triangles;
    vid_t*      d_off;
    gpu::allocate(d_triangles, hornet_init.nV() + 1);
    gpu::allocate(d_off,       hornet_init.nV() + 1);
    host::copyToDevice(hornet_init.csr_offsets(), nV + 1, d_off);

    auto              triNE = new int[nE];
    auto        h_triangles = new triangle_t[nV + 1];
    //int64_t allTrianglesCPU = 0;
    //int64_t       sumDevice = 0;

    float minTimecuStinger = 10e9;

    const int    blocksToTest = sizeof(arrayBlocks) / sizeof(int);
    const int blockSizeToTest = sizeof(arrayBlockSize) / sizeof(int);
    const int       tSPToTest = sizeof(arrayThreadPerIntersection) /sizeof(int);

    Timer<DEVICE, seconds> TM;

    for (int b = 0; b < blocksToTest; b++) {
        int blocks = arrayBlocks[b];
        for (int bs = 0; bs < blockSizeToTest; bs++) {
            int sps = arrayBlockSize[bs];
            for (int t = 0; t < tSPToTest; t++) {
                int     tsp = arrayThreadPerIntersection[t];
                int shifter = arrayThreadShift[t];
                int     nbl = sps / tsp;

                HornetGraph hornet_graph(hornet_init);
                hornet_graph.allocateEdgeDeletion(nE, CSR_WIDE |
                                         OUT_OF_PLACE | REMOVE_CROSS_DUPLICATE);
                KTruss kt(hornet_graph);

                if (alg & 1) {
                    kt.setInitParameters(tsp, nbl, shifter, blocks, sps);
                    kt.init();
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.run();

                    TM.stop();
                    std::cout << "graph=" << graph_name
                              << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;
                }
                if (alg & 2) {
                    kt.setInitParameters(tsp, nbl, shifter, blocks, sps);
                    kt.init();
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.runDynamic();

                    TM.stop();
                    std::cout << "graph=" << graph_name
                              << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;

                    if (TM.duration() < minTimecuStinger)
                        minTimecuStinger = TM.duration();
                }
                if (alg & 4) {
                    kt.setInitParameters(tsp, nbl, shifter, blocks, sps);
                    kt.init();
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.runForK(max_K);

                    TM.stop();

                    std::cout << "graph=" << graph_name
                              << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;
                }
                kt.release();
            }
        }
    }
    gpu::free(d_triangles);
    gpu::free(d_off);
    delete[] h_triangles;
    delete[] triNE;
}
