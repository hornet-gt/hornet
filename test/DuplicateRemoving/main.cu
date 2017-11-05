#include <algorithm>
#include <chrono>
#include <cub/cub.cuh>
#include <iostream>
#include <random>
#include <XLib.hpp>

using  edge_t = unsigned;
using edge2_t = typename xlib::make2_str<edge_t>::type;

void sortUniqueTest(edge2_t* batch, int batch_size, int V, bool debug = false);
void hashTableTest(edge2_t* batch, int batch_size, int V, bool debug = false);

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        ERROR("Syntax error:"
              "./duplicate <batch_size> <n_of_vertices> <debug:0/1>")
    }
    int batch_size, V, debug = false;
    try {
        batch_size = std::stoi(argv[1]);
        V          = std::stoi(argv[2]);
        if (argc == 4)
            debug  = std::stoi(argv[3]);
    } catch (const std::exception& ex) {
        ERROR("Syntax error")
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<edge_t> distribution(0, V - 1);

    /*edge_t test1[] = {7,6,1,1,4,2,1,2};
    edge_t test2[] = {4,5,5,7,3,7,5,7};
    batch_size = 8;*/

    auto batch = new edge2_t[batch_size];
    for (int i = 0; i < batch_size; i++) {
        //batch[i] = xlib::make2<edge_t>(test1[i], test2[i]);
        batch[i] = xlib::make2<edge_t>(distribution(generator),
                                       distribution(generator));
        if (debug)
            std::cout << batch[i] << "\n";
    }

    auto tmp_batch = new edge2_t[batch_size];
    std::copy(batch, batch + batch_size, tmp_batch);
    std::sort(tmp_batch, tmp_batch + batch_size);
    auto it = std::unique(tmp_batch, tmp_batch + batch_size);
    std::cout << "Host unique: " << std::distance(tmp_batch, it) << "\n\n";
    delete[] tmp_batch;

    sortUniqueTest(batch, batch_size, V, debug);
    hashTableTest(batch, batch_size, V, debug);

    delete[] batch;
}

#include "SortUnique.cu"
#include "HashTable.cu"
