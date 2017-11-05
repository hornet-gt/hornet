#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <XLib.hpp>

using  edge_t = unsigned;
using edge2_t = typename xlib::make2_str<edge_t>::type;

void detectTest(edge2_t* batch, int batch_size, int V, int threshold,
                bool debug = false);

int main(int argc, char* argv[]) {
    int batch_size, V, threshold, debug = false;
    if (argc != 4 && argc != 5) {
        ERROR("Syntax error:"
              "./detect <batch_size> <n_of_vertices> <threshold> <debug:0/1>")
    }
    try {
        batch_size = std::stoi(argv[1]);
        V          = std::stoi(argv[2]);
        threshold  = std::stoi(argv[3]);
        if (argc == 5)
            debug  = std::stoi(argv[4]);
    } catch (const std::exception& ex) {
        ERROR("Syntax error")
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<edge_t> distribution(0, V - 1);

    /*edge_t test1[] = {1,2,3,4,4,4,4,5};
    edge_t test2[] = {1,2,3,4,4,4,4,5};
    batch_size = sizeof(test1) / sizeof(edge_t);
    V = 6;
    threshold = 3;*/

    auto batch = new edge2_t[batch_size];
    for (int i = 0; i < batch_size; i++) {
        //batch[i] = xlib::make2<edge_t>(test1[i], test2[i]);
        batch[i] = xlib::make2<edge_t>(distribution(generator),
                                       distribution(generator));
        if (debug)
            std::cout << batch[i] << "\n";
    }

    auto h_histogram = new int[batch_size]();
    for (int i = 0; i < batch_size; i++)
        h_histogram[batch[i].x]++;
    int oversize = std::count_if(h_histogram, h_histogram + V,
                                 [&](int x){ return x > threshold; } );
    std::cout << "Host n. of oversize: " << oversize << "\n\n";
    delete[] h_histogram;

    detectTest(batch, batch_size, V, threshold, debug);

    delete[] batch;
}

#include "Detect.cu"
