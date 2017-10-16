#include "Hornet.hpp"
#include <GraphIO/GraphWeight.hpp>

/**
 * @brief Ensure that the GraphIO GraphWeight class can read weights
 * @author Kasimir Gabert <kasimir@gatech.edu>
 */
int main(int argc, char* argv[]) {
    graph::GraphWeight<int32_t, int32_t, float> graph;
    graph.read(argv[1]);
    graph.print();

    return 0;
}
