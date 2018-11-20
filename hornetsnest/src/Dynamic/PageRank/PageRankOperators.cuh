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

//------------------------------------------------------------------------------
#pragma once

namespace hornets_nest {
//namespace hornet_alg {
struct InitStreaming {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(vid_t src) {
        pd().visited[src]     = 0;
        pd().visitedDlt[src]  = 0;
        pd().diffPR[src]      = 0.0f;
        pd().delta[src]       = 0.0f;
        *pd().reduction_out   = 0;
    }
};
//------------------------------------------------------------------------------
struct SetupInsertions {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(Vertex& src_vertex, Vertex& dst_vertex) {
        auto src = src_vertex.id();
        auto dst = dst_vertex.id();
        //atomicAdd(kd().KC + src, kd().alpha);
        //atomicAdd(kd().new_paths_prev + src, 1);
        //vid_t prev = atomicCAS(kd().active + src, 0, kd().iteration);
        //if (prev == 0)
        //    kd().active_queue.insert(src);
    }
};
//------------------------------------------------------------------------------
struct SetupDeletions {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(Vertex& src_vertex, Vertex& dst_vertex) {
        //double minus_alpha = -kd().alpha;
        auto           src = src_vertex.id();
        auto           dst = dst_vertex.id();

        //atomicAdd(kd().KC + src, minus_alpha);
        //atomicAdd(kd().new_paths_prev + src, -1);
        //vid_t prev = atomicCAS(kd().active + src, 0, kd().iteration);
        //if (prev == 0)
        //    kd().active_queue.insert(src);
    }
};
//------------------------------------------------------------------------------
//struct RecomputeInsertionContriUndirected {
struct RecomputeContri {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(Vertex& src_vertex, Vertex& dst_vertex) {
        auto src = src_vertex.id();
        auto dst = dst_vertex.id();
        auto degree_src = src_vertex.degree();
        auto degree_dst = dst_vertex.degree();
        if (degree_dst == 0) return;

        //pd().curr_pr[src]  = pd().normalized_damp +
        //                           pd().damp * pd().curr_pr[src];

        float update_diff = pd().damp*(pd().prev_pr[dst]/degree_dst);//(pr->usedOld[dst])); //@@old dst
        float update_prop = pd().damp*(update_diff/degree_src);

        atomicAdd(pd().diffPR+src,update_diff);
        atomicAdd(pd().contri+src,update_diff); 

        if(fabs(update_prop) > pd().epsilon){
            if (pd().visited[src] == 0) {
                //CAS: old == compare ? val : old
                auto temp = pd().visited[src] + 1;
                auto old = atomicCAS(pd().visited+src,0,temp);
                if (old == 0) { 
                    pd().queue2.insert(src);  //pr->
                }   
            }
        }else{
            atomicAdd(pd().delta+src,update_diff);
            if ((pd().visited[src] == 0) && (pd().visitedDlt[src] == 0 )) {
                //CAS: old == compare ? val : old
                auto temp = pd().visitedDlt[src] + 1;
                auto old = atomicCAS(pd().visitedDlt+src,0,temp);
                if (old == 0) { 
                    pd().queueDlt.insert(src); 
                }   
            }
        } 
    }
};

//------------------------------------------------------------------------------
struct RecomputeDeletionContriUndirected {
//TO DO
};
//------------------------------------------------------------------------------

//struct UpdateDeltaAndMove {
struct UpdateDltMove {
    HostDeviceVar<PrDynamicData> pd;

    //OPERATOR(Vertex& vertex_src) {
      OPERATOR(vid_t vertex_src) {
        auto src = vertex_src;
        //auto src = vertex_src.id();
        if (pd().delta[src] > pd().epsilon)
        {
            if (pd().visited[src] == 0) {
                //CAS: old == compare ? val : old
                auto temp = pd().visited[src] + 1;
                auto old = atomicCAS(pd().visited+src,0,temp);
                if (old == 0) {
                    //prType delta = pr->delta[src]; //$$pair with recomputeContributionUndirected, updateContributionsUndirected
                    //atomicAdd(pr->contri+src,delta);
                    pd().delta[src] = 0.0;
                    pd().queue2.insert(src); 
                }   
            }
        }
    }
};
//------------------------------------------------------------------------------
//struct UpdateContributionAndCopy {
struct UpdateContriCopy {
    HostDeviceVar<PrDynamicData> pd;
#if 0
    OPERATOR(Vertex& src) {
        atomicAdd(pd().curr_pr + src.id(),
                  pd().contri[src.id()]);
    }
#else
    OPERATOR(vid_t src) {
        atomicAdd(pd().curr_pr + src, pd().contri[src]);
    }
#endif
};

//------------------------------------------------------------------------------
//struct UpdateDeltaAndCopy {
struct UpdateDltCopy {
    HostDeviceVar<PrDynamicData> pd;
#if 0
    OPERATOR(Vertex& src) {
        atomicAdd(pd().curr_pr + src.id(),
                  pd().contri[src.id()]);
    }
#else
    OPERATOR(vid_t src) {
        atomicAdd(pd().curr_pr + src, pd().contri[src]);
    }
#endif
};

//------------------------------------------------------------------------------
//struct UpdateContributionsUndirected {
struct UpdateContri {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto src = vertex.id();
        auto dst = edge.dst_id();
        auto dstVertex = edge.dst();
        //auto degree_src = src.degree();
        auto degree_src = vertex.degree();
        //auto degree_dst = dst.degree();
        auto degree_dst = dstVertex.degree(); //????????
 
        if (degree_src == 0) return;

        //pd().curr_pr[src]  = pd().normalized_damp +
        //                           pd().damp * pd().curr_pr[src];

        float update_diff = pd().damp*((pd().prev_pr[src]/degree_src) - (pd().prev_pr[src]/degree_src));//(pr->usedOld[src]));
        float update_prop = pd().damp*(update_diff/degree_dst);

        //atomicAdd(pd().diffPR+src,update_diff);
        atomicAdd(pd().contri+dst,update_diff); 

        if(fabs(update_prop) > pd().epsilon){
            if (pd().visited[dst] == 0) {
                //CAS: old == compare ? val : old
                auto temp = pd().visited[dst] + 1;
                auto old = atomicCAS(pd().visited+dst,0,temp);
                if (old == 0) { 
                    //pr->queue2.insert(dst); 
                    pd().queue2.insert(dst);
                }   
            }
        }else{
            atomicAdd(pd().delta+dst,update_diff);
            if ((pd().visited[dst] == 0) && (pd().visitedDlt[dst] == 0 )) {
                //CAS: old == compare ? val : old
                auto temp = pd().visitedDlt[dst] + 1;
                auto old = atomicCAS(pd().visitedDlt+dst,0,temp);
                if (old == 0) { 
                    pd().queueDlt.insert(dst); 
                }   
            }
        } 
    }
};
//------------------------------------------------------------------------------
//struct RemoveContributionsUndirected {
struct RemoveContri {
    HostDeviceVar<PrDynamicData> pd;
#if 0
    OPERATOR(Vertex& src) {
        float diffPR = pd().diffPR[src];
        atomicAdd(pd().curr_pr + src.id(),-diffPR);
    }
#else
    OPERATOR(vid_t src) {
        float diffPR = pd().diffPR[src];
        atomicAdd(pd().curr_pr + src, -diffPR);
    }
#endif
};
//------------------------------------------------------------------------------
//struct PrevEqualCurr {
struct PrevEqCurr {
    HostDeviceVar<PrDynamicData> pd;
#if 0
    OPERATOR(Vertex& src) {
        pd().prev_pr[src]  = pd().curr_pr[src];
    }
#else
    OPERATOR(vid_t src) {
        atomicAdd(pd().prev_pr + src, pd().curr_pr[src]);
    }
#endif
};
//------------------------------------------------------------------------------

struct ResetCurr {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(vid_t src) {
        pd().curr_pr[src]     = 0.0f;
        *(pd().reduction_out) = 0;
    }
};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//------------------------------------------------------------------------------

struct ComputeContribuitionPerVertex {
    HostDeviceVar<PrDynamicData> pd;
#if 1
    OPERATOR(Vertex& vertex_src) {
        auto degree = vertex_src.degree();
        auto    src = vertex_src.id();
        pd().contri[src] = degree == 0 ? 0.0f :
                                  pd().prev_pr[src] / degree;
    }
#else
    OPERATOR(vid_t vertex_src) {
        auto degree = vertex_src.degree();
        auto src = vertex_src;
        atomicAdd(pd().curr_pr + src, pd().contri[src]);
    }
#endif
};

//------------------------------------------------------------------------------

struct AddContribuitions {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(Vertex& src, Edge& edge) {
        atomicAdd(pd().curr_pr + edge.dst_id(),
                  pd().contri[src.id()]);
    }
};

//------------------------------------------------------------------------------

struct AddContribuitionsUndirected {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(Vertex& src_vertex, Edge& edge) {
        atomicAdd(pd().curr_pr+ src_vertex.id(),
                  pd().contri[edge.dst_id()]);
    }
};

//------------------------------------------------------------------------------

struct DampAndDiffAndCopy {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(vid_t src) {
        // pd().curr_pr[src]=(1-pd().damp)/float(pd().nV)+
        //                          pd().damp*pd().curr_pr[src];
        pd().curr_pr[src]  = pd().normalized_damp +
                                   pd().damp * pd().curr_pr[src];

        pd().abs_diff[src] = fabsf(pd().curr_pr[src] -
                                         pd().prev_pr[src]);
        pd().prev_pr[src]  = pd().curr_pr[src];
    }
};

//------------------------------------------------------------------------------

struct Sum {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(vid_t src) {
        atomicAdd(pd().reduction_out, pd().abs_diff[src]);
    }
};

//------------------------------------------------------------------------------

struct SumPr {
    HostDeviceVar<PrDynamicData> pd;

    OPERATOR(vid_t src) {
        atomicAdd(pd().reduction_out, pd().prev_pr[src] );
    }
};

//------------------------------------------------------------------------------

struct SetIds {
    vid_t* ids;

    OPERATOR(vid_t src) {
        ids[src] = src;
    }
};
//-------------------------------------------------------------------------------
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

} // hornetAlgs namespace

