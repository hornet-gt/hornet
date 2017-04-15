/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */

#include "Util/BatchFunctions.hpp"
#include "Support/Numeric.hpp"
#include <utility>
#include <chrono>
#include <random>

void generateInsertBatch(id_t* batch_src, id_t* batch_dest,
                         int batch_size, const graph::GraphStd<>& graph,
                         BatchProperty prop) {

    if (!prop.weighted) {
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<id_t> distribution(0, graph.nV() - 1);
        for (int i = 0; i < batch_size; i++) {
            batch_src[i]  = distribution(gen);
            batch_dest[i] = distribution(gen);
        }
    }
    else {
        xlib::WeightedRandomGenerator<id_t>
            weighted_gen(graph.out_degrees_array(), graph.nV());
        for (int i = 0; i < batch_size; i++) {
            batch_src[i]  = weighted_gen.get();
            batch_dest[i] = weighted_gen.get();
        }
    }

    if (prop.print || prop.sort) {
        auto tmp_batch = new std::pair<id_t, id_t>[batch_size];
        for (int i = 0; i < batch_size; i++)
            tmp_batch[i] = std::make_pair(batch_src[i], batch_dest[i]);

        std::sort(tmp_batch, tmp_batch + batch_size);
        if (prop.sort) {
            for (int i = 0; i < batch_size; i++) {
                batch_src[i]  = tmp_batch[i].first;
                batch_dest[i] = tmp_batch[i].second;
            }
        }
        if (prop.print) {
            std::cout << "Batch:\n";
            for (int i = 0; i < batch_size; i++) {
                std::cout << "(" << tmp_batch[i].first << ","
                          << tmp_batch[i].second << ")\n";
            }
            std::cout << std::endl;
        }
        delete[] tmp_batch;
    }
}
