/**
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>
 *   ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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

Please cite:
A. van der Grinten, E. Bergamini, O. Green, H. Meyerhenke, D. Bader, 
“Scalable Katz Ranking Computation in Large Dynamic Graphs”, 
European Symposium on Algorithms, Helsinki, Finland, 2018 
*/

#include "Static/KatzCentrality/Katz.cuh"
#include "KatzOperators.cuh"

using length_t = int;

namespace hornets_nest {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

KatzCentrality::KatzCentrality(HornetGraph& hornet, int max_iteration, int K,
                               int max_degree, bool is_static) :
                                       StaticAlgorithm(hornet),
                                       load_balancing(hornet),
                                       is_static(is_static) {
    if (max_iteration <= 0)
        ERROR("Number of max iterations should be greater than zero")

    hd_katzdata().nV            = hornet.nV();
    hd_katzdata().K             = K;
    hd_katzdata().max_degree    = max_degree;
    hd_katzdata().alpha         = 1.0 / (static_cast<double>(max_degree) + 1.0);
    hd_katzdata().max_iteration = max_iteration;

    auto nV = hornet.nV();

    if (is_static) {
        gpu::allocate(hd_katzdata().num_paths_data, nV * 2);
        hd_katzdata().num_paths_prev = hd_katzdata().num_paths_data;
        hd_katzdata().num_paths_curr = hd_katzdata().num_paths_data + nV;
        hd_katzdata().num_paths      = nullptr;
        h_paths_ptr                  = nullptr;
    }
    else {
        gpu::allocate(hd_katzdata().num_paths_data, nV * max_iteration);
        gpu::allocate(hd_katzdata().num_paths, max_iteration);

        host::allocate(h_paths_ptr, max_iteration);
        for(int i = 0; i < max_iteration; i++)
            h_paths_ptr[i] = hd_katzdata().num_paths_data + nV * i;

        hd_katzdata().num_paths_prev = h_paths_ptr[0];
        hd_katzdata().num_paths_curr = h_paths_ptr[1];
        host::copyToDevice(h_paths_ptr, max_iteration, hd_katzdata().num_paths);
    }
    gpu::allocate(hd_katzdata().KC,          nV);
    gpu::allocate(hd_katzdata().lower_bound, nV);
    gpu::allocate(hd_katzdata().upper_bound, nV);

    gpu::allocate(hd_katzdata().is_active,             nV);
    gpu::allocate(hd_katzdata().vertex_array_sorted,   nV);
    gpu::allocate(hd_katzdata().vertex_array_unsorted, nV);
    gpu::allocate(hd_katzdata().lower_bound_sorted,    nV);
    gpu::allocate(hd_katzdata().lower_bound_unsorted,  nV);

    reset();
}

KatzCentrality::~KatzCentrality() {
    release();
}

void KatzCentrality::reset() {
    hd_katzdata().iteration = 1;

    if (is_static) {
        hd_katzdata().num_paths_prev = hd_katzdata().num_paths_data;
        hd_katzdata().num_paths_curr = hd_katzdata().num_paths_data +
                                       hornet.nV();
    }
    else {
        hd_katzdata().num_paths_prev = h_paths_ptr[0];
        hd_katzdata().num_paths_curr = h_paths_ptr[1];
    }
}

void KatzCentrality::release(){
    gpu::free(hd_katzdata().num_paths_data);
    gpu::free(hd_katzdata().num_paths);
    gpu::free(hd_katzdata().KC);
    gpu::free(hd_katzdata().lower_bound);
    gpu::free(hd_katzdata().upper_bound);
    gpu::free(hd_katzdata().vertex_array_sorted);
    gpu::free(hd_katzdata().vertex_array_unsorted);
    gpu::free(hd_katzdata().lower_bound_sorted);
    gpu::free(hd_katzdata().lower_bound_unsorted);
    host::free(h_paths_ptr);
}

void KatzCentrality::run() {
    forAllnumV(hornet, Init { hd_katzdata });

    hd_katzdata().iteration  = 1;
    hd_katzdata().num_active = hornet.nV();

    while (hd_katzdata().num_active > hd_katzdata().K &&
           hd_katzdata().iteration < hd_katzdata().max_iteration) {

        hd_katzdata().alphaI            = std::pow(hd_katzdata().alpha,
                                                   hd_katzdata().iteration);
        hd_katzdata().lower_bound_const = std::pow(hd_katzdata().alpha,
                                                  hd_katzdata().iteration + 1) /
                                        (1.0 - hd_katzdata().alpha);
        hd_katzdata().upper_bound_const = std::pow(hd_katzdata().alpha,
                                                  hd_katzdata().iteration + 1) /
                                        (1.0 - hd_katzdata().alpha *
                                 static_cast<double>(hd_katzdata().max_degree));
        hd_katzdata().num_active = 0; // Each iteration the number of active
                                     // vertices is set to zero.

        forAllnumV (hornet, InitNumPathsPerIteration { hd_katzdata } );
        forAllEdges(hornet, UpdatePathCount          { hd_katzdata },
                    load_balancing);
        forAllnumV (hornet, UpdateKatzAndBounds      { hd_katzdata } );

        hd_katzdata.sync();

        hd_katzdata().iteration++;
        if(is_static) {
            std::swap(hd_katzdata().num_paths_curr,
                      hd_katzdata().num_paths_prev);
        }
        else {
            auto                    iter = hd_katzdata().iteration;
            hd_katzdata().num_paths_prev = h_paths_ptr[iter - 1];
            hd_katzdata().num_paths_curr = h_paths_ptr[iter - 0];
        }
        auto         old_active_count = hd_katzdata().num_active;
        hd_katzdata().num_prev_active = hd_katzdata().num_active;
        hd_katzdata().num_active      = 0; // Resetting active vertices for
                                           // sorting

        // Notice that the sorts the vertices in an incremental order based on
        // the lower bounds.
        // The algorithms requires the vertices to be sorted in an decremental
        // fashion.
        // As such, we use the num_prev_active variables to store the number of
        // previous active vertices and are able to find the K-th from last
        // vertex (which is essentially going from the tail of the array).
        xlib::CubSortByKey<double, vid_t>::srun
            (hd_katzdata().lower_bound_unsorted,
             hd_katzdata().vertex_array_unsorted,
             old_active_count, hd_katzdata().lower_bound_sorted,
             hd_katzdata().vertex_array_sorted);

        forAllnumV(hornet, CountActive { hd_katzdata } );
        hd_katzdata.sync();
    }
}

void KatzCentrality::copyKCToHost(double* d) {
    gpu::copyToHost(hd_katzdata().KC, hornet.nV(), d);
}

// This function should only be used directly within run() and is currently
// commented out due to to large execution overheads.
void KatzCentrality::printKMostImportant() {
    ulong_t* num_paths_curr;
    ulong_t* num_paths_prev;
    int*     vertex_array;
    int*     vertex_array_unsorted;
    double*  KC;
    double*  lower_bound;
    double*  upper_bound;

    auto nV = hornet.nV();
    host::allocate(num_paths_curr, nV);
    host::allocate(num_paths_prev, nV);
    host::allocate(vertex_array,   nV);
    host::allocate(vertex_array_unsorted, nV);
    host::allocate(KC,          nV);
    host::allocate(lower_bound, nV);
    host::allocate(upper_bound, nV);

    gpu::copyToHost(hd_katzdata().lower_bound, nV, lower_bound);
    gpu::copyToHost(hd_katzdata().upper_bound, nV, upper_bound);
    gpu::copyToHost(hd_katzdata().KC, nV, KC);
    gpu::copyToHost(hd_katzdata().vertex_array_sorted, nV, vertex_array);
    gpu::copyToHost(hd_katzdata().vertex_array_unsorted, nV,
                    vertex_array_unsorted);

    if (hd_katzdata().num_prev_active > hd_katzdata().K) {
        for (int i = hd_katzdata().num_prev_active - 1;
                i >= hd_katzdata().num_prev_active - hd_katzdata().K; i--) {
            vid_t j = vertex_array[i];
            std::cout << j << "\t\t" << KC[j] << "\t\t" << upper_bound[j]
                      << upper_bound[j] - lower_bound[j] << "\n";
        }
    }
    std::cout << std::endl;

    host::free(num_paths_curr);
    host::free(num_paths_prev);
    host::free(vertex_array);
    host::free(vertex_array_unsorted);
    host::free(KC);
    host::free(lower_bound);
    host::free(upper_bound);
}

int KatzCentrality::get_iteration_count() {
    return hd_katzdata().iteration;
}

bool KatzCentrality::validate() {
    return true;
}

//KatzData KatzCentrality::katz_data() {
//    return hd_katzdata;
//}
//
} // namespace hornets_nest
