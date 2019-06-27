/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "Util/BatchFunctions.hpp"
#include "Host/Numeric.hpp"
#include <chrono>
#include <random>
#include <utility>

namespace hornets_nest {

BatchGenProperty::BatchGenProperty(const detail::BatchGenEnum& obj) noexcept :
             xlib::PropertyClass<detail::BatchGenEnum, BatchGenProperty>(obj) {}

//------------------------------------------------------------------------------

void generateBatch(const graph::GraphStd<>& graph, int& batch_size,
                   vert_t* batch_src, vert_t* batch_dst,
                   const BatchGenType& batch_type,
                   const BatchGenProperty& prop) {
    using vid_distribution = std::uniform_int_distribution<vert_t>;

    if (batch_type == BatchGenType::REMOVE) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                    .count();
        std::mt19937_64 gen(seed);
        vid_distribution distribution_src(0, graph.nV() - 1);
        for (int i = 0; i < batch_size; i++) {
            auto      src = distribution_src(gen);
            if (graph.out_degree(src) == 0) {
                i--;
                continue;
            }
            vid_distribution distribution_dst(0, graph.out_degree(src) - 1);
            auto    index = distribution_dst(gen);
            batch_src[i]  = src;
            batch_dst[i] = graph.vertex(src).neighbor_id(index);
        }
    }
    else if (prop == batch_gen_property::WEIGHTED) {
        xlib::WeightedRandomGenerator<vert_t>
            weighted_gen(graph.out_degrees_ptr(), graph.nV());
        for (int i = 0; i < batch_size; i++) {
            batch_src[i]  = weighted_gen.get();
            batch_dst[i] = weighted_gen.get();
        }
    }
    else {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                    .count();
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<vert_t> distribution(0, graph.nV() - 1);
        for (int i = 0; i < batch_size; i++) {
            batch_src[i]  = distribution(gen);
            batch_dst[i] = distribution(gen);
        }
    }

    if (prop == batch_gen_property::PRINT || prop == batch_gen_property::UNIQUE) {
        auto tmp_batch = new std::pair<vert_t, vert_t>[batch_size];
        for (int i = 0; i < batch_size; i++)
            tmp_batch[i] = std::make_pair(batch_src[i], batch_dst[i]);

        std::sort(tmp_batch, tmp_batch + batch_size);
        if (prop == batch_gen_property::UNIQUE) {
            auto    it = std::unique(tmp_batch, tmp_batch + batch_size);
            batch_size = std::distance(tmp_batch, it);
            for (int i = 0; i < batch_size; i++) {
                batch_src[i] = tmp_batch[i].first;
                batch_dst[i] = tmp_batch[i].second;
            }
        }
        if (prop == batch_gen_property::PRINT) {
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

} // namespace hornets_nest
