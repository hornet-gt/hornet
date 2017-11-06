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
            int bin = 0;
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
        printf("number %d of edges: %d\n", i, hd_queue_info().queue_sizes[i]);

    // Next, add each edge into the correct corresponding queue
    for (auto i = 0; i < MAX_ADJ_UNIONS_BINS; i++)
        hd_queue_info().queues[i].initialize((size_t)hd_queue_info().queue_sizes[0]+1);
    forAllEdgesSrcDst(hornet, bin_edges {hd_queue_info, false}, load_balancing);

    // Phase 2: run the operator on each queued edge as appropriate
    for (auto i = 0; i < MAX_ADJ_UNIONS_BINS; i++) {
        hd_queue_info().queues[i].swap();
        size_t threads_per = 0;
        // FIXME: change Operator and its args as well
        if (i == 0) {
            threads_per = 8;
        } else if (i == 1) {
            threads_per = 32;
        }
        forAllEdgesAdjUnion(hd_queue_info().queues[i], op, threads_per);
    }
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
