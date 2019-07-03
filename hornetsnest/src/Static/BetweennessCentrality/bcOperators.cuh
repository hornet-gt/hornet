/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>       
 *   ogreen@nvidia.com
 *  @author Muhammad Osama Sakhi                                            <br>
 *   Georgia Institute of Technology                                        <br>       
 * @date July, 2018
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

#include "Static/BetweennessCentrality/bc.cuh"

namespace hornets_nest {


// Used at the very beginning of every BC computation.
// Used only once.
struct InitBC {
    HostDeviceVar<BCData> bcd;


    OPERATOR(vid_t src) {
        bcd().bc[src] = 0.0;
    }
};

struct InitRootData {
    HostDeviceVar<BCData> bcd;


    OPERATOR(int  i) {
        bcd().d[bcd().root]=0;
        bcd().sigma[bcd().root]=1.0;

    }
};
 


// Used at the very beginning of every BC computation.
// Once per root
struct InitOneTree {
    HostDeviceVar<BCData> bcd;

    // Used at the very beginning
    OPERATOR(vid_t src) {
        bcd().d[src] = INT32_MAX;
        bcd().sigma[src] = 0;
        bcd().delta[src] = 0.0;
    }
};

struct BC_BFSTopDown {
    HostDeviceVar<BCData> bcd;

    OPERATOR(Vertex& src, Edge& edge){
        vid_t v = src.id(), w = edge.dst_id();        
        degree_t nextLevel = bcd().d[v] + 1;

        degree_t prev = atomicCAS(bcd().d + w, INT32_MAX, nextLevel);
        if (prev == INT32_MAX) {
            bcd().queue.insert(w);
        }
        if (bcd().d[w] == nextLevel) {
            atomicAdd(bcd().sigma + w, bcd().sigma[v]);
        }
    }
};


struct BC_DepAccumulation {
    HostDeviceVar<BCData> bcd;

    OPERATOR(Vertex& src, Edge& edge){

        vid_t v = src.id(), w = edge.dst_id();        

        degree_t *d = bcd().d;  // depth
        paths_t *sigma = bcd().sigma;
        bc_t *delta = bcd().delta;

        if (d[w] == (d[v] + 1))
        {   
            atomicAdd(delta + v, ((bc_t) sigma[v] / (bc_t) sigma[w]) * (1.0 + delta[w]));
        }
    }
};


// Used at the very beginning of every BC computation.
// Once per root
struct IncrementBC {
    HostDeviceVar<BCData> bcd;

    // Used at the very beginning
    OPERATOR(vid_t src) {
        if(src != bcd().root)
            bcd().bc[src]+=bcd().delta[src];
    }
};

// Used at the very beginning of every BC computation.
// Once per root
struct IncrementBCNew {
    HostDeviceVar<BCData> bcd;

    // Used at the very beginning
    OPERATOR(Vertex& sr) {
        vid_t src = sr.id();
        if(src != bcd().root)
            bcd().bc[src]+=bcd().delta[src];
    }   
};

} // namespace hornets_nest
