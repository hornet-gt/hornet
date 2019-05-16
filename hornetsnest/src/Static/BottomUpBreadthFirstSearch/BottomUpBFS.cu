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

struct countDegrees {
int32_t *bins;

OPERATOR(Vertex& vertex) {

    __shared__ int32_t localBins[33];
    int id = threadIdx.x;
    if(id==0){
        for (int i=0; i<33; i++)
           localBins[i]=0;
    }

   
    __syncthreads();

    int32_t size = vertex.degree();
    int32_t myBin = __clz(size);

    int32_t my_pos = atomicAdd(localBins+myBin, 1);

    __syncthreads();

    if(id==0){
         for (int i=0; i<33; i++){
             atomicAdd(bins+i, localBins[i]);
         }
    }
} 
};

__global__ void binPrefixKernel(int32_t *bins, int32_t *d_binsPrefix){

int i = threadIdx.x + blockIdx.x *blockDim.x;
    if(i>=1)
        return;
    d_binsPrefix[0]=0;
    for(int b=0; b<33; b++){
        d_binsPrefix[b+1]=d_binsPrefix[b]+bins[b];
    }
}

template<typename HornetDevice> __global__ void  rebinKernel(
  HornetDevice hornet ,
  const vid_t    *original,
  int32_t    *d_binsPrefix,
  vid_t     *d_reOrg,
  int N){
    int i = threadIdx.x + blockIdx.x *blockDim.x;
    __shared__ int32_t localBins[33];
    __shared__ int32_t localPos[33];

    int id = threadIdx.x;

    if(id<33){
      localBins[id]=0;
      localPos[id]=0;
    }
    __syncthreads();

    int myBin,myPos;

    if(i<N){
        int32_t adjSize= hornet.vertex(original[i]).degree();
        
        myBin  = __clz(adjSize);
        myPos = atomicAdd(localBins+myBin, 1);
    }
    __syncthreads();

    if(id<33){
        localPos[id]=atomicAdd(d_binsPrefix+id, localBins[id]);
    }
  __syncthreads();

    if(i<N){

        int pos = localPos[myBin]+myPos;
        d_reOrg[pos]=original[i];

    }
}

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



template<typename HornetDevice, typename Operator>
__global__
void forAllVerticesKernel1(HornetDevice              hornet,
                           vid_t                    *vertices_array,
                           int                       num_items,
                           Operator                  op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (vid_t i = id; i < num_items; i += stride) {
        auto vertex = hornet.vertex(vertices_array[i]);
        op(vertex);
    }
}

template<typename HornetDevice, typename Operator>
__global__
void forAllVerticesKernel2(HornetDevice              hornet,
                           const vid_t               *vertices_array,
                           int                       num_items,
                           Operator                  op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (vid_t i = id; i < num_items; i += stride) {
        auto vertex = hornet.vertex(vertices_array[i]);
        op(vertex);
    }
}


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
    gpu::allocate(d_edges, hornet.nV());
    gpu::allocate(d_edges_frontier, hornet.nV());
    gpu::allocate(d_binsPrefix, hornet.nV());
    gpu::allocate(d_lrbRelabled, hornet.nV());
    gpu::allocate(d_bin, 33);
    reset();
}

BfsBottomUp2::~BfsBottomUp2() {
    gpu::free(d_distances);
    gpu::free(d_edges);
    gpu::free(d_edges_frontier);
    gpu::free(d_binsPrefix);
    gpu::free(d_lrbRelabled);
}

void BfsBottomUp2::reset() {
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    auto edges = d_edges;
    auto edges_frontier = d_edges_frontier;
    forAllnumV(hornet, [=] __device__ (int i){ distances[i] = INF; } );
    forAllnumV(hornet, [=] __device__ (int i){ edges[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ edges_frontier[i] = 0; } );
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
                forAllVertices(hornet_in, queue_inf, countDegrees { d_bin });
        
                binPrefixKernel<<< xlib::ceil_div<BLOCK_SIZE_OP2>(queue_inf.size()), BLOCK_SIZE_OP2 >>>(d_bin, d_binsPrefix);
                rebinKernel<<< xlib::ceil_div<BLOCK_SIZE_OP2>(queue_inf.size()), BLOCK_SIZE_OP2 >>>(hornet_in.device_side(),queue_inf.device_input_ptr(), d_binsPrefix,d_lrbRelabled,queue_inf.size());
            }

            if(queue.size() > 0){

                if(bu_flag){
                   
                    forAllVertices(hornet_in, queue_inf, vertexBFSOperator { current_level, d_distances});
                    forAllVertices(hornet_in, queue_inf, OPERATOR_AddToQueue { current_level, d_distances, queue, queue_inf });

                }else{

                  //no vertex relabeling
                  forAllVerticesKernel2<<< xlib::ceil_div<BLOCK_SIZE_OP2>(queue_inf.size()), BLOCK_SIZE_OP2 >>>(hornet_in.device_side(),queue_inf.device_input_ptr(), queue_inf.size() , vertexBFSOperator { current_level, d_distances});
                  forAllVerticesKernel2<<< xlib::ceil_div<BLOCK_SIZE_OP2>(queue_inf.size()), BLOCK_SIZE_OP2 >>>(hornet_in.device_side(),queue_inf.device_input_ptr(), queue_inf.size() , OPERATOR_AddToQueue { current_level, d_distances, queue, queue_inf });

                }

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
    gpu::free(d_edges);
    gpu::free(d_edges_frontier);
    gpu::free(d_binsPrefix);
    gpu::free(d_lrbRelabled);
    gpu::free(d_bin);

    d_distances = nullptr;
    d_edges = nullptr;
    d_edges_frontier = nullptr;
    d_binsPrefix = nullptr;
    d_lrbRelabled = nullptr;
    d_bin = nullptr;
}

bool BfsBottomUp2::validate() {
    std::cout << "\nTotal enqueue vertices: "
              << xlib::format(queue.enqueue_items())
              << std::endl;

    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
                                  hornet.csr_edges(), hornet.nE());


    BFS<vid_t, eoff_t> bfs(graph);
    bfs.run(bfs_source);

    auto h_distances = bfs.result();


    return gpu::equal(h_distances, h_distances + graph.nV(), d_distances);
}

} // namespace hornets_nest
