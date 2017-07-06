#include "Static/KTruss/KTruss.cuh"
#include "Support/Device/Timer.cuh"

using namespace timer;
using namespace custinger;
using namespace custinger_alg;

void runKtruss(const cuStingerInit& custinger_init, int alg, int maxk,
              const std::string& graphName);

int main(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT);

    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_ptr(),
                                 graph.out_edges_ptr());

    int alg = 3, maxk = 3;
    if (argc >= 3)
        alg = std::stoi(argv[2]);
    if (argc >= 4)
        maxk = std::stoi(argv[3]);

    auto weights = new int[graph.nE()]();
    custinger_init.insertEdgeData(weights);
    runKtruss(custinger_init, alg, maxk, graph.name());
    delete[] weights;
}

//==============================================================================

const int                arrayBlocks[] = { 16000 };
const int             arrayBlockSize[] = { 32 };
const int arrayThreadPerIntersection[] = { 1 };
const int           arrayThreadShift[] = { 0 };

void runKtruss(const cuStingerInit& custinger_init, int alg, int maxk,
               const std::string& graphName) {

    int nv = custinger_init.nV();
    int ne = custinger_init.nE();

    triangle_t* d_triangles;
    vid_t* d_off;
    cudaMalloc(&d_triangles, (nv + 1) * sizeof(triangle_t));
    cudaMalloc(&d_off, (nv + 1) * sizeof(vid_t));
    cuMemcpyToDevice(custinger_init.csr_offsets(), nv + 1, d_off);


    auto              triNE = new int[ne];
    auto        h_triangles = new triangle_t[nv + 1];
    //int64_t allTrianglesCPU = 0;
    //int64_t       sumDevice = 0;

    float minTimecuStinger = 10e9;

    const int    blocksToTest = sizeof(arrayBlocks) / sizeof(int);
    const int blockSizeToTest = sizeof(arrayBlockSize) / sizeof(int);
    const int       tSPToTest = sizeof(arrayThreadPerIntersection) /sizeof(int);
    BatchUpdate batch_update(ne);

    Timer<DEVICE, seconds> TM;

    for (int b = 0; b < blocksToTest; b++) {
        int blocks = arrayBlocks[b];
        for (int bs = 0; bs < blockSizeToTest; bs++) {
            int sps = arrayBlockSize[bs];
            for (int t = 0; t < tSPToTest; t++) {
                int     tsp = arrayThreadPerIntersection[t];
                int shifter = arrayThreadShift[t];
                int     nbl = sps / tsp;

                cuStinger custiger_graph(custinger_init);
                KTruss kt(custiger_graph, batch_update);

                if (alg & 1) {
                    kt.setInitParameters(nv,ne, tsp, nbl, shifter, blocks, sps);
                    kt.init();
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.run();

                    TM.stop();
                    std::cout << "graph=" << graphName << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;

                    //custiger_graph.freecuStinger();
                }
                if (alg & 2) {
                    kt.setInitParameters(nv, ne, tsp, nbl, shifter,
                                         blocks, sps);
                    kt.init();
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.runDynamic();

                    TM.stop();
                    std::cout << "graph=" << graphName << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;
                    //kt.release();

                    if (TM.duration() < minTimecuStinger)
                        minTimecuStinger = TM.duration();
                    //custing2.freecuStinger();
                }
                if (alg & 4) {
                    kt.setInitParameters(nv, ne, tsp, nbl, shifter,
                                         blocks, sps);
                    kt.init();
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.runForK(maxk);

                    TM.stop();

                    std::cout << "graph=" << graphName << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;
                    //kt.release();
                    //custiger_graph.freecuStinger();
                }
                kt.release();
            }
        }
    }
    cuFree(d_triangles, d_off);
    delete[] h_triangles;
    delete[] triNE;
}
