/*
Please cite:
* J. Fox, O. Green, K. Gabert, X. An, D. Bader, “Fast and Adaptive List Intersections on the GPU”, 
IEEE High Performance Extreme Computing Conference (HPEC), 
Waltham, Massachusetts, 2018
* O. Green, J. Fox, A. Tripathy, A. Watkins, K. Gabert, E. Kim, X. An, K. Aatish, D. Bader, 
“Logarithmic Radix Binning and Vectorized Triangle Counting”, 
IEEE High Performance Extreme Computing Conference (HPEC), 
Waltham, Massachusetts, 2018
* O. Green, P. Yalamanchili ,L.M. Munguia, “Fast Triangle Counting on GPU”, 
Irregular Applications: Architectures and Algorithms (IA3), 
New Orleans, Louisiana, 2014 
*/



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
    triangle_t* d_triPerVertex;

    OPERATOR(Vertex& v1, Vertex& v2, int flag) {
        triangle_t count = 0;
        int deg1 = v1.degree();
        int deg2 = v2.degree();
        vid_t* ui_begin = v1.neighbor_ptr();
        vid_t* vi_begin = v2.neighbor_ptr();
        vid_t* ui_end = ui_begin+deg1-1;
        vid_t* vi_end = vi_begin+deg2-1;
        int comp_equals, comp1, comp2;
        while (vi_begin <= vi_end && ui_begin <= ui_end) {
            comp_equals = (*ui_begin == *vi_begin);
            count += comp_equals;
            comp1 = (*ui_begin >= *vi_begin);
            comp2 = (*ui_begin <= *vi_begin);
            vi_begin += comp1;
            ui_begin += comp2;
            // early termination
            if ((vi_begin > vi_end) || (ui_begin > ui_end))
                break;
        }
        atomicAdd(d_triPerVertex+v1.id(), count);
        atomicAdd(d_triPerVertex+v2.id(), count);
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
        //atomicAdd(d_triPerVertex+v.id(), count);
    }
};

void TriangleCounting2::copyTCToHost(triangle_t* h_tcs) {
    gpu::copyToHost(triPerVertex, hornet.nV(), h_tcs);
}

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

void TriangleCounting2::run() {
    //printf("Inside run()\n");
    forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerVertex }, 1);
    //forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCount { triPerVertex });
}

void TriangleCounting2::run(const int WORK_FACTOR=1){
    forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerVertex }, WORK_FACTOR);
}


void TriangleCounting2::release(){
    //printf("Inside release\n");
    gpu::free(triPerVertex);
    triPerVertex = nullptr;
}

void TriangleCounting2::init(){
    //printf("Inside init. Printing hornet.nV(): %d\n", hornet.nV());
    gpu::allocate(triPerVertex, hornet.nV());
    reset();
}

} // namespace hornets_nest
