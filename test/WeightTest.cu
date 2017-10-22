#include "Hornet.hpp"
#include <GraphIO/GraphWeight.hpp>

using weight_t = float;

using namespace hornets_nest;

/**
 * @brief Ensure that the GraphIO GraphWeight class can read weights
 * @author Kasimir Gabert <kasimir@gatech.edu>
 */
int main(int argc, char* argv[]) {
    graph::GraphWeight<vid_t, eoff_t, weight_t> graph;
    graph.read(argv[1]);
    graph.print();
}
