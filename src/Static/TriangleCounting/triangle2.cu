
#include <cuda.h>
#include <cuda_runtime.h>

#include "Static/TriangleCounting/triangle2.cuh"

namespace hornets_nest {

TriangleCounting2::TriangleCounting2(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet)

{                                       
}

TriangleCounting2::~TriangleCounting2(){
    release();
}

struct OPERATOR_InitTriangleCounts {
    triangle_t *d_triPerVertex;

    OPERATOR (Vertex &vertex) {
        d_triPerVertex[vertex.id()] = 0;
    }
};

/*
 * Naive intersection operator
 * Assumption: access to entire adjacencies of v1 and v2 required
 */
struct OPERATOR_AdjIntersectionCount {
    int* d_triPerVertex;

    OPERATOR(Vertex& v1, Vertex& v2) {
        int equalCount = 0;
        int outDeg1 = v1.out_degree();
        int outDeg2 = v2.out_degree();
        int eoff1 = 0;
        int eoff2 = 0;
        vid_t vid_curr1;
        vid_t vid_curr2;
        int comp;
        while (eoff1 < outDeg1 && eoff2 < outDeg2) {
            vid_curr1 = v1.edge(eoff1).dst_id();
            vid_curr2 = v2.edge(eoff2).dst_id();
            comp = vid_curr1 - vid_curr2;
            equalCount += (comp == 0);
            eoff1 += (comp <= 0 && !0);
            eoff2 += (comp >= 0 && !0);
        }
        atomicAdd(d_triPerVertex+v1.id(), equalCount);
        atomicAdd(d_triPerVertex+v2.id(), equalCount);
    }
};


struct OPERATOR_AdjIntersectionCountBalanced {
    triangle_t* d_triPerVertex;

    OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, vid_t* vi_begin, vid_t* vi_end, int FLAG) {
        int count = 0;
        if (!FLAG) {
            int comp_equals, comp1, comp2, ui_bound, vi_bound;
            //printf("Intersecting %d, %d: %d -> %d, %d -> %d\n", u.id(), v.id(), *ui_begin, *ui_end, *vi_begin, *vi_end);
            while (vi_begin <= vi_end && ui_begin <= ui_end) {
                comp_equals = (*ui_begin == *vi_begin);
                count += comp_equals;
                comp1 = (*ui_begin >= *vi_begin);
                comp2 = (*ui_begin <= *vi_begin);
                ui_bound = (ui_begin == ui_end);
                vi_bound = (vi_begin == vi_end);
                // early termination
                if ((ui_bound && comp2) || (vi_bound && comp1))
                    break;
                if ((comp1 && !vi_bound) || ui_bound)
                    vi_begin += 1;
                if ((comp2 && !ui_bound) || vi_bound)
                    ui_begin += 1;
            }
        } else {
            vid_t vi_low, vi_high, vi_mid;
            while (ui_begin <= ui_end) {
                auto search_val = *ui_begin;
                vi_low = 0;
                vi_high = vi_end-vi_begin;
                while (vi_low <= vi_high) {
                    vi_mid = (vi_low+vi_high)/2;
                    auto comp = (*(vi_begin+vi_mid) - search_val);
                    if (!comp) {
                        count += 1;
                        break;
                    }
                    if (comp > 0) {
                        vi_high = vi_mid-1;
                    } else if (comp < 0) {
                        vi_low = vi_mid+1;
                    }
                }
                ui_begin += 1;
            }
        }

        atomicAdd(d_triPerVertex+u.id(), count);
        atomicAdd(d_triPerVertex+v.id(), count);
    }
};


triangle_t TriangleCounting2::countTriangles(){

    triangle_t* h_triPerVertex;
    host::allocate(h_triPerVertex, hornet.nV());
    gpu::copyToHost(triPerVertex, hornet.nV(), h_triPerVertex);
    triangle_t sum=0;
    for(int i=0; i<hornet.nV(); i++){
        // printf("%d %ld\n", i,outputArray[i]);
        sum+=h_triPerVertex[i];
    }
    free(h_triPerVertex);
    //triangle_t sum=gpu::reduce(hd_triangleData().triPerVertex, hd_triangleData().nv+1);

    return sum;
}


void TriangleCounting2::reset(){
    //printf("Inside reset()\n");
    forAllVertices(hornet, OPERATOR_InitTriangleCounts { triPerVertex });
}

void TriangleCounting2::run(){
    //printf("Inside run()\n");
    forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerVertex });
}


void TriangleCounting2::release(){
    //printf("Inside release\n");
    gpu::free(triPerVertex);
    triPerVertex = nullptr;
}

void TriangleCounting2::init(){
    //printf("Inside init()\n");
    gpu::allocate(triPerVertex, hornet.nV());
    reset();
}

} // namespace hornets_nest
