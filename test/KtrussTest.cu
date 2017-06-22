#include "Static/KTruss/Ktruss.cuh"

using namespace timer;
using namespace custinger;
using namespace custinger_alg;

int runKtruss(const cuStingerInit& custinger_init, int alg, int maxk,
              const std::string& graphName);

int main(int argc, char *argv[]) {
    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);

    cuStingerInit custinger_init(graph.nV(), graph.nE(), graph.out_offsets(),
                                 graph.out_edges());

    int alg = 3, maxk = 3;
    if (argc >= 3)
        alg = std::stoi(argv[2]);
    if (argc >= 4)
        maxk = std::stoi(argv[3]);

    runKtruss(custinger_init, alg, maxk, graph.name());
}

//==============================================================================

const int                arrayBlocks[] = { 16000 };
const int             arrayBlockSize[] = { 32 };
const int arrayThreadPerIntersection[] = { 1 };
const int           arrayThreadShift[] = { 0 };

int runKtruss(const cuStingerInit& custinger_init, int alg, int maxk,
              const std::string& graphName) {

    triangle_t* d_triangles;
    cuMalloc(d_triangles, nv + 1);

    auto              triNE = new int[ne];
    auto        h_triangles = new triangle_t[nv + 1];
    int64_t allTrianglesCPU = 0;
    int64_t       sumDevice = 0;

    float minTimecuStinger = 10e9;

    const int    blocksToTest = sizeof(arrayBlocks) / sizeof(int);
    const int blockSizeToTest = sizeof(arrayBlockSize) / sizeof(int);
    const int       tSPToTest = sizeof(arrayThreadPerIntersection) /sizeof(int);

    for (int b = 0; b < blocksToTest; b++) {
        int blocks = arrayBlocks[b];
        for (int bs = 0; bs < blockSizeToTest; bs++) {
            int sps = arrayBlockSize[bs];
            for (int t = 0; t < tSPToTest; t++) {
                int     tsp = arrayThreadPerIntersection[t];
                int shifter = arrayThreadShift[t];
                int     nbl = sps / tsp;

                cuStinger custiger_graph(custinger_init);
                kTruss kt;

                if (alg & 1) {
                    kt.setInitParameters(nv,ne, tsp, nbl, shifter, blocks, sps);
                    kt.init(custinger);
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.run(custinger);

                    TM.stop();
                    std::cout << "graph=" << graphName << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;
                    kt.release();
                    custinger.freecuStinger();
                }
                if (alg & 2) {
                    kt2.setInitParameters(nv, ne, tsp, nbl, shifter,
                                          blocks, sps);
                    kt2.init(custing2);
                    kt2.copyOffsetArrayDevice(d_off);
                    kt2.reset();

                    TM.start();

                    kt2.runDynamic(custing2);

                    TM.stop();
                    std::cout << "graph=" << graphName << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;
                    kt2.release();

                    if (TM.duration() < minTimecuStinger)
                        minTimecuStinger = TM.duration();
                    //custing2.freecuStinger();
                }
                if (alg & 4) {
                    kt.setInitParameters(nv, ne, tsp, nbl, shifter,
                                         blocks, sps);
                    kt.init(custinger);
                    kt.copyOffsetArrayDevice(d_off);
                    kt.reset();

                    TM.start();

                    kt.runForK(custinger, maxk);

                    TM.stop();

                    std::cout << "graph=" << graphName << "\nk=" << kt.getMaxK()
                              << ":" << TM.duration() << std::endl;
                    kt.release();
                    //custinger.freecuStinger();
                }
            }
        }
    }
    cuFree(d_triangles);
    delete[] h_triangles;
    delete[] triNE;
}
