/**
 * @brief Brim test program
 * @file
 */
//#include "Static/ShortestPath/SSSP.cuh"
#include <GraphIO/GraphWeight.hpp>
#include <GraphIO/Brim.hpp>
#include <BasicTypes.hpp>
#include <Device/Timer.cuh>

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace graph;
    using namespace hornets_nest;

    graph::GraphWeight<vid_t, eoff_t, int> graph;
    graph.read(argv[1]);

    Brim<vid_t, eoff_t, int> brim(graph);

    Timer<DEVICE> TM;
    TM.start();

    brim.run();

    TM.stop();
    TM.print("Brim");

    std::cout << "MPG Check: " << brim.check() << std::endl;
	brim.check_from_file(argv[2]);

    /*auto is_correct = brim.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return is_correct;*/
}
