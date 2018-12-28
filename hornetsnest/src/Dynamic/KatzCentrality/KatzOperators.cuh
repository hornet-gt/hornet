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

// Used only once when the streaming katz data structure is initialized
struct InitStreaming {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(vid_t vertex_id) {
        kd().new_paths_curr[vertex_id] = 0;
        kd().new_paths_prev[vertex_id] = kd().num_paths[1][vertex_id];
        kd().active[vertex_id]         = 0;
    }
};

//------------------------------------------------------------------------------

struct SetupInsertions {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(Vertex& src_vertex, Vertex& dst_vertex) {
        auto src = src_vertex.id();
        auto dst = dst_vertex.id();
        atomicAdd(kd().KC + src, kd().alpha);
        atomicAdd(kd().new_paths_prev + src, 1);
        vid_t prev = atomicCAS(kd().active + src, 0, kd().iteration);
        if (prev == 0)
            kd().active_queue.insert(src);
    }
};

//------------------------------------------------------------------------------

struct SetupDeletions {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(Vertex& src_vertex, Vertex& dst_vertex) {
        double minus_alpha = -kd().alpha;
        auto src = src_vertex.id();
        auto dst = dst_vertex.id();

        atomicAdd(kd().KC + src, minus_alpha);
        //atomicAdd(kd().new_paths_prev + src, -1);
        atomicAdd(kd().new_paths_prev + src, static_cast<std::remove_pointer_t<decltype(kd().new_paths_prev)>>(-1));//atomicSub or atmoicDec will be clearer but these functions currently (up to CUDA 10) do not support 64 bit integer. It may be better to store the old value and assert the old value is larger than 0 for safety.
        vid_t prev = atomicCAS(kd().active + src, 0, kd().iteration);
        if (prev == 0)
            kd().active_queue.insert(src);
    }
};

//------------------------------------------------------------------------------

struct InitActiveNewPaths {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(vid_t vertex_id) {
        auto npath = kd().num_paths[kd().iteration][vertex_id];
        kd().new_paths_curr[vertex_id] = npath;
    }
};

//------------------------------------------------------------------------------

struct FindNextActive {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        auto src = vertex.id();

        vid_t prev = atomicCAS(kd().active + dst, 0, kd().iteration);
        if (prev == 0) {
            kd().active_queue.insert(dst);
            kd().new_paths_curr[dst]= kd().num_paths[kd().iteration][dst];
        }
    }
};

//------------------------------------------------------------------------------

struct UpdateActiveNewPaths {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        auto src = vertex.id();

        if(kd().active[src] < kd().iteration){
            ulong_t val_to_add = kd().new_paths_prev[src] -
                                 kd().num_paths[kd().iteration - 1][src];
            atomicAdd(kd().new_paths_curr+dst, val_to_add);
        }
    }
};

//------------------------------------------------------------------------------

struct UpdateNewPathsBatchInsert {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(Vertex& src_vertex, Vertex& dst_vertex) {
        auto src = src_vertex.id();
        auto dst = dst_vertex.id();

        ulong_t val_to_add = kd().num_paths[kd().iteration - 1][dst];
        atomicAdd(kd().new_paths_curr + src, val_to_add);
    }
};

//------------------------------------------------------------------------------

struct UpdateNewPathsBatchDelete {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(Vertex& src_vertex, Vertex& dst_vertex) {
        auto src = src_vertex.id();
        auto dst = dst_vertex.id();

        ulong_t val_to_remove = -kd().num_paths[kd().iteration - 1][dst];
        atomicAdd(kd().new_paths_curr + src, val_to_remove);
    }
};

//------------------------------------------------------------------------------

struct UpdatePrevWithCurr {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(vid_t vertex_id) {
        // Note the conversion to signed long long int!! Especially important
        //for edge deletions where this diff can be negative
        long long int paths_diff = kd().new_paths_curr[vertex_id] -
                                   kd().num_paths[kd().iteration][vertex_id];

        kd().KC[vertex_id] += kd().alphaI * paths_diff;
        if(kd().active[vertex_id] < kd().iteration) {
            auto prev = kd().new_paths_prev[vertex_id];
            kd().num_paths[kd().iteration - 1][vertex_id] = prev;
        }
        kd().new_paths_prev[vertex_id] = kd().new_paths_curr[vertex_id];
    }
};

//------------------------------------------------------------------------------

struct UpdateLastIteration {
    HostDeviceVar<KatzDynamicData> kd;

    OPERATOR(vid_t vertex_id) {
        if (kd().active[vertex_id] < kd().iteration) {
            auto prev = kd().new_paths_prev[vertex_id];
            kd().num_paths[kd().iteration - 1][vertex_id] = prev;
        }
    }
};

} // namespace hornets_nest
