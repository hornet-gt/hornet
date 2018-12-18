#include <iostream>
#include <XLib.hpp>

int main(int argc, char* argv[]) {
    using namespace graph;
    using namespace xlib;
    using namespace timer;

    GraphStd<> graph;
    graph.read(argv[1]);
    //graph.print();
    //graph.print_raw();
    Bitmask bitmask(graph.nV());
    xlib::Queue<int> queue(graph.nV());
    auto distance = new int[graph.nV()];
    queue.insert(0);
    bitmask[0] = true;
    distance[0] = 0;
    Timer<HOST> TM;
    TM.start();

    while (!queue.is_empty()) {
        auto current_id = queue.extract();
        for (auto e : graph.get_vertex(current_id)) {
            auto dest = e.dest().id();
            if (!bitmask[dest]) {
                bitmask[dest] = true;
                distance[dest] = distance[current_id] + 1;
                queue.insert(dest);
            }
        }
    }
    TM.stop();
    TM.print("iterator");

    bitmask.clear();
    queue.clear();
    queue.insert(0);
    bitmask[0] = true;
    distance[0] = 0;
    const auto offsets = graph.out_offsets_array();
    const auto   edges = graph.out_edges_array();

    TM.start();

    while (!queue.is_empty()) {
        auto i = queue.extract();
        for (int j = offsets[i]; j < offsets[i + 1]; j++) {
            auto dest = edges[j];
            if (!bitmask[dest]) {
                bitmask[dest] = true;
                distance[dest] = distance[i] + 1;
                queue.insert(dest);
            }
        }
    }

    TM.stop();
    TM.print("raw");
}
