namespace hornets_nest {
namespace detail {

template<typename Operator>
__global__ void forAllKernel(int size, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (auto i = id; i < size; i += stride)
        op(i);
}

template<typename T, typename Operator>
__global__ void forAllKernel(T* __restrict__ array, int size, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (auto i = id; i < size; i += stride) {
        auto value = array[i];
        op(value);
    }
}

template<typename HornetDevice, typename T, typename Operator>
__global__ void forAllEdgesAdjUnionSequentialKernel(HornetDevice hornet, T* __restrict__ array, int size, int flag, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (auto i = id; i < size; i += stride) {
        auto src_vtx = hornet.vertex(array[i].x);
        auto dst_vtx = hornet.vertex(array[i].y);
        auto src_adj_iter = src_vtx.edge_begin();
        auto dst_adj_iter = dst_vtx.edge_begin();
        auto src_adj_end = src_vtx.edge_end();
        auto dst_adj_end = dst_vtx.edge_end();
        op(src_adj_iter, src_adj_end, dst_adj_iter, dst_adj_end, flag);
    }
}

namespace adj_union {
    __device__ __forceinline__
    void initialize(degree_t diag_id,
                    degree_t u_len,
                    degree_t v_len,
                    vid_t* __restrict__ u_min,
                    vid_t* __restrict__ u_max,
                    vid_t* __restrict__ v_min,
                    vid_t* __restrict__ v_max,
                    int*   __restrict__ found) {
        if (diag_id == 0) {
            *u_min = *u_max = *v_min = *v_max = 0;
            *found = 1;
        }
        else if (diag_id < u_len) {
            *u_min = 0;
            *u_max = diag_id;
            *v_max = diag_id;
            *v_min = 0;
        }
        else if (diag_id < v_len) {
            *u_min = 0;
            *u_max = u_len;
            *v_max = diag_id;
            *v_min = diag_id - u_len;
        }
        else {
            *u_min = diag_id - v_len;
            *u_max = u_len;
            *v_min = diag_id - u_len;
            *v_max = v_len;
        }
    }

    __device__ __forceinline__
    void workPerThread(degree_t uLength,
                    degree_t vLength,
                    int threadsPerIntersection,
                    int threadId,
                    int* __restrict__ outWorkPerThread,
                    int* __restrict__ outDiagonalId) {
    int      totalWork = uLength + vLength;
    int  remainderWork = totalWork % threadsPerIntersection;
    int  workPerThread = totalWork / threadsPerIntersection;

    int longDiagonals  = threadId > remainderWork ? remainderWork : threadId;
    int shortDiagonals = threadId > remainderWork ? threadId - remainderWork : 0;

    *outDiagonalId     = (workPerThread + 1) * longDiagonals +
                            workPerThread * shortDiagonals;
    *outWorkPerThread  = workPerThread + (threadId < remainderWork);
    }

    __device__ __forceinline__
    void bSearch(unsigned found,
                degree_t    diagonalId,
                const vid_t*  __restrict__ uNodes,
                const vid_t*  __restrict__ vNodes,
                const degree_t*  __restrict__ uLength,
                vid_t* __restrict__ outUMin,
                vid_t* __restrict__ outUMax,
                vid_t* __restrict__ outVMin,
                vid_t* __restrict__ outVMax,
                vid_t* __restrict__ outUCurr,
                vid_t* __restrict__ outVCurr) {
        vid_t length;
        while (!found){
            *outUCurr = (*outUMin + *outUMax) >> 1;
            *outVCurr = diagonalId - *outUCurr;
            if (*outVCurr >= *outVMax){
                length = *outUMax - *outUMin;
                if (length == 1){
                    found = 1;
                    continue;
                }
            }

            unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
            unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
            if (comp1 && !comp2)
                found = 1;
            else if (comp1){
                *outVMin = *outVCurr;
                *outUMax = *outUCurr;
            }
            else{
                *outVMax = *outVCurr;
                *outUMin = *outUCurr;
            }
        }

        if (*outVCurr >= *outVMax && length == 1 && *outVCurr > 0 &&
                *outUCurr > 0 && *outUCurr < *uLength - 1)
        {
            unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
            unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
            if (!comp1 && !comp2)
            {
                (*outUCurr)++;
                (*outVCurr)--;
            }
        }
    }

    __device__ __forceinline__
    void indexBinarySearch(vid_t* data, vid_t arrLen, vid_t key, int& pos) {
        int low = 0;
        int high = arrLen - 1;
        while (high >= low)
        {
            int middle = (low + high) / 2;
            if (data[middle] == key)
            {
                pos = middle;
                return;
            }
            if (data[middle] < key)
                low = middle + 1;
            if (data[middle] > key)
                high = middle - 1;
        }
    }
}

template<typename HornetDevice, typename T, typename Operator>
__global__ void forAllEdgesAdjUnionBalancedKernel(HornetDevice hornet, T* __restrict__ array, int size, size_t threads_per_union, int flag, Operator op) {
    using namespace adj_union;
    int       id = blockIdx.x * blockDim.x + threadIdx.x;
    int queue_id = id / threads_per_union;
    int thread_id = id % threads_per_union;
    int stride = blockDim.x * gridDim.x;
    int queue_stride = stride / threads_per_union;
    for (auto i = queue_id; i < size; i += queue_stride) {
        auto src_vtx = hornet.vertex(array[i].x);
        auto dst_vtx = hornet.vertex(array[i].y);
        int u_len = src_vtx.out_degree();
        int v_len = dst_vtx.out_degree();
        
        // Find the work required per thread
        int work_per_thread, diag_id;
        workPerThread(u_len, v_len, threads_per_block, id,
                    &work_per_thread, &diag_id);

        int       work_index = 0;
        int            found = 0;
        vid_t u_min, u_max, v_min, v_max, u_curr, v_curr;
        // firstFound logic
        __shared__ vid_t firstFound[1024];
        int tId = threadIdx.x % threads_per_union // ~
        firstFound[(queue_id*threads_per_union)+tId] = 0; // ~ check

        if (work_per_thread > 0) {
            // For the binary search, we are figuring out the initial poT of search.
            initialize(diag_id, u_len, v_len, &u_min, &u_max,
                    &v_min, &v_max, &found);
            u_curr = 0;
            v_curr = 0;

            bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max,
                    &v_min, &v_max, &u_curr, &v_curr);

            op(u_curr, u_len, v_curr, v_len, flag);
        }

        printf("thread %d - on edge %p %p\n", thread_id, src_adj_iter, dst_adj_iter);
    }
}

template<typename Operator>
__global__ void forAllnumVKernel(vid_t d_nV, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (auto i = id; i < d_nV; i += stride)
        op(i);
}

template<typename Operator>
__global__ void forAllnumEKernel(eoff_t d_nE, Operator op) {
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    int  stride = gridDim.x * blockDim.x;

    for (eoff_t i = id; i < d_nE; i += stride)
        op(i);
}

template<typename HornetDevice, typename Operator>
__global__ void forAllVerticesKernel(HornetDevice hornet,
                                     Operator     op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (vid_t i = id; i < hornet.nV(); i += stride) {
        auto vertex = hornet.vertex(i);
        op(vertex);
    }
}

template<typename HornetDevice, typename Operator>
__global__
void forAllVerticesKernel(HornetDevice              hornet,
                          const vid_t* __restrict__ vertices_array,
                          int                       num_items,
                          Operator                  op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (vid_t i = id; i < num_items; i += stride) {
        auto vertex = hornet.vertex(vertices_array[i]);
        op(vertex);
    }
}
/*
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK,
         typename HornetDevice, typename Operator>
__global__
void forAllEdgesKernel(const eoff_t* __restrict__ csr_offsets,
                       HornetDevice               hornet,
                       Operator                   op) {

    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    const auto lambda = [&](int pos, degree_t offset) {
                                auto vertex = hornet.vertex(pos);
                                op(vertex, vertex.edge(offset));
                            };
    xlib::binarySearchLB<BLOCK_SIZE>(csr_offsets, hornet.nV() + 1,
                                     smem, lambda);
}*/

} //namespace detail

//==============================================================================
//==============================================================================
// stub
#define MAX_ADJ_UNIONS_BINS 2
namespace adj_unions {
    struct queue_info {
        int queue_sizes[MAX_ADJ_UNIONS_BINS];
        TwoLevelQueue<vid2_t> queues[MAX_ADJ_UNIONS_BINS];
    };

    struct bin_edges {
        HostDeviceVar<queue_info> d_queue_info;
        bool countOnly;
        OPERATOR(Vertex& src, Vertex& dst, Edge& edge) {
            // Choose the bin to place this edge into
            if (src.id() > dst.id()) return; // imposes ordering
            int bin = 1;
            if (src.degree() + dst.degree() > 128)
                bin = 1;

            // Either count or add the item to the appropriate queue
            if (countOnly)
                atomicAdd(&(d_queue_info.ptr()->queue_sizes[bin]), 1);
            else
                d_queue_info().queues[bin].insert({ src.id(), edge.dst_id() });
        }
    };
}


template<typename HornetClass, typename Operator>
void forAllAdjUnions(HornetClass&         hornet,
                     const Operator&      op)
{
    using namespace adj_unions;
    HostDeviceVar<queue_info> hd_queue_info;

    load_balancing::VertexBased1 load_balancing ( hornet );

    // Initialize queue sizes to zero
    for (auto i = 0; i < MAX_ADJ_UNIONS_BINS; i++)
        hd_queue_info().queue_sizes[i] = 0;

    // Phase 1: determine and bin all edges based on edge neighbor properties
    // First, count the number to avoid creating excessive queues
    forAllEdgesSrcDst(hornet, bin_edges {hd_queue_info, true}, load_balancing);
    hd_queue_info.sync();

    for (auto i = 0; i < MAX_ADJ_UNIONS_BINS; i++)
        printf("queue=%d number of edges: %d\n", i, hd_queue_info().queue_sizes[i]);

    // Next, add each edge into the correct corresponding queue
    for (auto i = 0; i < MAX_ADJ_UNIONS_BINS; i++)
        hd_queue_info().queues[i].initialize((size_t)hd_queue_info().queue_sizes[i]+1);
    forAllEdgesSrcDst(hornet, bin_edges {hd_queue_info, false}, load_balancing);

    // Phase 2: run the operator on each queued edge as appropriate
    for (auto i = 0; i < MAX_ADJ_UNIONS_BINS; i++) {
        hd_queue_info().queues[i].swap();
        size_t threads_per = 0;
        int flag = 0;
        // FIXME: change Operator and its args as well
        if (hd_queue_info().queue_sizes[i] == 0) continue;
        if (i == 0) {
            threads_per = 1;
            forAllEdgesAdjUnionSequential(hornet, hd_queue_info().queues[i], op, flag);
        } else if (i == 1) {
            threads_per = 8;
            forAllEdgesAdjUnionBalanced(hornet, hd_queue_info().queues[i], op, threads_per, flag);
        } else if (i == 2) {
            // Imbalance case, flag = 1
            flag = 1;
            //forAllEdgesAdjUnionSequential(hd_queue_info().queues[i], op, threads_per, flag);
        }
    }
}


template<typename HornetClass, typename Operator>
void forAllEdgesAdjUnionSequential(HornetClass &hornet, TwoLevelQueue<vid2_t> queue, const Operator &op, int flag) {
    auto size = queue.size();
    if (size == 0)
        return;
    detail::forAllEdgesAdjUnionSequentialKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (hornet.device_side(), queue.device_input_ptr(), size, flag, op);
    CHECK_CUDA_ERROR
}

template<typename HornetClass, typename Operator>
void forAllEdgesAdjUnionBalanced(HornetClass &hornet, TwoLevelQueue<vid2_t> queue, const Operator &op, size_t threads_per_union, int flag) {
    auto size = queue.size();
    if (size == 0)
        return;
    detail::forAllEdgesAdjUnionBalancedKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size*threads_per_union), BLOCK_SIZE_OP2 >>>
        (hornet.device_side(), queue.device_input_ptr(), size, threads_per_union, flag, op);
    CHECK_CUDA_ERROR
}

template<typename Operator>
void forAll(size_t size, const Operator& op) {
    if (size == 0)
        return;
    detail::forAllKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (size, op);
    CHECK_CUDA_ERROR
}

template<typename T, typename Operator>
void forAll(const TwoLevelQueue<T>& queue, const Operator& op) {
    auto size = queue.size();
    if (size == 0)
        return;
    detail::forAllKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (queue.device_input_ptr(), size, op);
    CHECK_CUDA_ERROR
}

//------------------------------------------------------------------------------

template<typename HornetClass, typename Operator>
void forAllnumV(HornetClass& hornet, const Operator& op) {
    detail::forAllnumVKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(hornet.nV()), BLOCK_SIZE_OP2 >>>
        (hornet.nV(), op);
    CHECK_CUDA_ERROR
}

//------------------------------------------------------------------------------

template<typename HornetClass, typename Operator>
void forAllnumE(HornetClass& hornet, const Operator& op) {
    detail::forAllnumEKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(hornet.nE()), BLOCK_SIZE_OP2 >>>
        (hornet.nE(), op);
    CHECK_CUDA_ERROR
}

//==============================================================================

template<typename HornetClass, typename Operator>
void forAllVertices(HornetClass& hornet, const Operator& op) {
    detail::forAllVerticesKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(hornet.nV()), BLOCK_SIZE_OP2 >>>
        (hornet.device_side(), op);
    CHECK_CUDA_ERROR
}

//------------------------------------------------------------------------------

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&         hornet,
                 const Operator&      op,
                 const LoadBalancing& load_balancing) {
    const int PARTITION_SIZE = xlib::SMemPerBlock<BLOCK_SIZE_OP2, vid_t>::value;
    int num_partitions = xlib::ceil_div<PARTITION_SIZE>(hornet.nE());

    load_balancing.apply(hornet, op);
}

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdgesSrcDst(HornetClass&         hornet,
                       const Operator&      op,
                       const LoadBalancing& load_balancing) {
    const int PARTITION_SIZE = xlib::SMemPerBlock<BLOCK_SIZE_OP2, vid_t>::value;
    int num_partitions = xlib::ceil_div<PARTITION_SIZE>(hornet.nE());

    load_balancing.applySrcDst(hornet, op);
}

//==============================================================================

template<typename HornetClass, typename Operator, typename T>
void forAllVertices(HornetClass&    hornet,
                    const vid_t*    vertex_array,
                    int             size,
                    const Operator& op) {
    detail::forAllVerticesKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (hornet.device_side(), vertex_array, size, op);
    CHECK_CUDA_ERROR
}

template<typename HornetClass, typename Operator>
void forAllVertices(HornetClass&                hornet,
                    const TwoLevelQueue<vid_t>& queue,
                    const Operator&             op) {
    auto size = queue.size();
    detail::forAllVerticesKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (hornet.device_side(), queue.device_input_ptr(), size, op);
    CHECK_CUDA_ERROR
}

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&    hornet,
                 const vid_t*    vertex_array,
                 int             size,
                 const Operator& op,
                 const LoadBalancing& load_balancing) {
    load_balancing.apply(hornet, vertex_array, size, op);
}
/*
template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass& hornet,
                 const TwoLevelQueue<vid_t>& queue,
                 const Operator& op, const LoadBalancing& load_balancing) {
    load_balancing.apply(hornet, queue.device_input_ptr(),
                        queue.size(), op);
    //queue.kernel_after();
}*/

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&                hornet,
                 const TwoLevelQueue<vid_t>& queue,
                 const Operator&             op,
                 const LoadBalancing&        load_balancing) {
    load_balancing.apply(hornet, queue.device_input_ptr(), queue.size(), op);
}

} // namespace hornets_nest
