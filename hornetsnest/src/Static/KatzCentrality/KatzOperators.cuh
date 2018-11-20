/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
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
 *
 * @file
 */
namespace hornets_nest {

struct Init {
    HostDeviceVar<KatzData> kd;

    // Used at the very beginning
    OPERATOR(vid_t src) {
        kd().num_paths_prev[src] = 1;
        kd().num_paths_curr[src] = 0;
        kd().KC[src]             = 0.0;
        kd().is_active[src]      = true;
    }
};

//------------------------------------------------------------------------------

struct InitNumPathsPerIteration {
    HostDeviceVar<KatzData> kd;

    OPERATOR(vid_t src) {
        kd().num_paths_curr[src] = 0;
    }
};

//------------------------------------------------------------------------------

struct UpdatePathCount {
    HostDeviceVar<KatzData> kd;

    OPERATOR(Vertex& src, Edge& edge){
        auto src_id = src.id();
        auto dst_id = edge.dst_id();
        atomicAdd(kd().num_paths_curr + src_id,
                  kd().num_paths_prev[dst_id]);
    }
};

//------------------------------------------------------------------------------

struct UpdateKatzAndBounds {
    HostDeviceVar<KatzData> kd;

    OPERATOR(vid_t src) {
        kd().KC[src] = kd().KC[src] + kd().alphaI *
                        static_cast<double>(kd().num_paths_curr[src]);
        kd().lower_bound[src] = kd().KC[src] + kd().lower_bound_const *
                                static_cast<double>(kd().num_paths_curr[src]);
        kd().upper_bound[src] = kd().KC[src] + kd().upper_bound_const *
                                static_cast<double>(kd().num_paths_curr[src]);

        if (kd().is_active[src]) {
            int pos = atomicAdd(&(kd.ptr()->num_active), 1);
            kd().vertex_array_unsorted[pos] = src;
            kd().lower_bound_unsorted[pos]  = kd().lower_bound[src];
        }
    }
};

//------------------------------------------------------------------------------

struct CountActive {
    HostDeviceVar<KatzData> kd;

    OPERATOR(vid_t src) {
        auto index = kd().vertex_array_sorted[kd().num_prev_active - kd().K];
        if (kd().upper_bound[src] > kd().lower_bound[index])
            atomicAdd(&(kd.ptr()->num_active), 1);
        else
            kd().is_active[src] = false;
    }
};

} // namespace hornets_nest
