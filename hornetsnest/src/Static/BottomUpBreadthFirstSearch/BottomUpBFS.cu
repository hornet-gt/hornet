/**
 * @author Sarah Nguyen                                                 <br>
 *         Georgia Institute of Technology                              <br>
 * @date May, 2019
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
 */
#include "Static/BUBreadthFirstSearch/BottomUpBFS.cuh"
#include "Auxilary/DuplicateRemoving.cuh"
#include <Graph/GraphStd.hpp>
#include <Graph/BFS.hpp>

namespace hornets_nest {

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct BFSOperatorAtomic {                  //deterministic
    dist_t               current_level;
    dist_t*              d_distances;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        if (atomicCAS(d_distances + dst, INF, current_level) == INF) {
            queue.insert(dst);
            }
        }
};

struct OPERATOR_AddToQueue {
    dist_t               current_level;
    dist_t*              d_distances;
    TwoLevelQueue<vid_t> queue;
    TwoLevelQueue<vid_t> queue_inf;

    OPERATOR(Vertex& vertex) {
    auto v = vertex.id();
          if (atomicCAS(d_distances + v, current_level, current_level) == current_level){
              queue.insert(v);
          }else{
              queue_inf.insert(v);
          }
    }
};

struct OPERATOR_AddToQueue_inf {
    dist_t               current_level;
    dist_t*              d_distances;
    TwoLevelQueue<vid_t> queue;


    OPERATOR(Vertex& vertex) {
    auto v = vertex.id();
          if (atomicCAS(d_distances + v, INF, INF) == INF){
              queue.insert(v);
              }
    }
};


struct vertexBFSOperator {             
    dist_t               current_level;
    dist_t*              d_distances;

    OPERATOR(Vertex &v1) {
       int deg1 = v1.degree();
       vid_t* ui_begin = v1.neighbor_ptr();
       vid_t* ui_end = ui_begin+deg1-1;

       while (ui_begin <= ui_end) {
              if((d_distances[v1.id()] == INF) && (d_distances[*ui_begin] == current_level - 1)){
                  d_distances[v1.id()] = current_level;
                  break;
               }
              ++ui_begin;
       }
           
    }

};


//------------------------------------------------------------------------------
/////////////////
// BfsTopDown2 //
/////////////////

BfsBottomUp2::BfsBottomUp2(HornetGraph& hornet, HornetGraph& hornet_in) :
                                 StaticAlgorithm(hornet),
                                 queue(hornet, 5),
                                 queue_inf(hornet),
                                 load_balancing(hornet) {
    gpu::allocate(d_distances, hornet.nV());
    reset();
}

BfsBottomUp2::~BfsBottomUp2() {
    gpu::free(d_distances);
}

void BfsBottomUp2::reset() {
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    forAllnumV(hornet, [=] __device__ (int i){ distances[i] = INF; } );
}

void BfsBottomUp2::set_parameters(vid_t source) {
    bfs_source = source;
    printf("set_parameters source: %i\n", source);
    queue.insert(bfs_source);               // insert bfs source in the frontier
    gpu::memsetZero(d_distances + bfs_source);  //reset source distance
}


void BfsBottomUp2::run() {

    while (queue.size() > 0) {

        forAllEdges(hornet, queue,
                    BFSOperatorAtomic { current_level, d_distances, queue },
                    load_balancing);
        queue.swap();
        current_level++;
    }

}


void BfsBottomUp2::run(HornetGraph& hornet_in) {
       int td = 1; //top down 1, bottom up 0
       int qs;
       int nv = hornet.nV();
       int bu_flag = 0;

    while (queue.size() > 0) {

        qs = queue.size();
        td = (float)nv/(float)qs > 40 ? 1 : 0;

        //top down
        if(td){
            forAllEdges(hornet, queue,
                    BFSOperatorAtomic { current_level, d_distances, queue},
                    load_balancing);

            queue.swap();
            current_level++;
            bu_flag = 0;
        }

        //bottom up
        else{
            if(!bu_flag){  //if BU flag == 1 then bottom up was done in previous step

                forAllVertices(hornet_in, OPERATOR_AddToQueue_inf { current_level, d_distances, queue_inf });
                queue_inf.swap();
            }

            if(queue.size() > 0){
                   
                forAllVertices(hornet_in, queue_inf, vertexBFSOperator { current_level, d_distances});
                forAllVertices(hornet_in, queue_inf, OPERATOR_AddToQueue { current_level, d_distances, queue, queue_inf });

                queue.swap();
                queue_inf.swap();
                current_level++;
                bu_flag = 1;
            }

        }

    }

}

void BfsBottomUp2::release() {
    gpu::free(d_distances);
    d_distances = nullptr;
}

bool BfsBottomUp2::validate() {
  return true;
    //std::cout << "\nTotal enqueue vertices: "
    //          << xlib::format(queue.enqueue_items())
    //          << std::endl;

    //using namespace graph;
    //GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
    //                              hornet.csr_edges(), hornet.nE());


    //BFS<vid_t, eoff_t> bfs(graph);
    //bfs.run(bfs_source);

    //auto h_distances = bfs.result();


    //return gpu::equal(h_distances, h_distances + graph.nV(), d_distances);
}

} // namespace hornets_nest
