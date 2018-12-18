#include <Graph/GraphStd.hpp>

int main(int argc, char* argv[]) {
    if (argc != 2)
        ERROR("Syntax: ", argv[0], " <graph_path>")

    graph::GraphStd<> graph(graph::structure_prop::ENABLE_INGOING, argv[1],
                            graph::parsing_prop::PRINT_INFO);
    graph.write_analysis("dataset.txt");
    graph.print_analysis();
}
