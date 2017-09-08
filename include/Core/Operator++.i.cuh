namespace hornet_alg {
namespace detail {

template<typename Operator>
__global__ void forAllKernel(int size, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < size; i += stride)
        op(i);
}

template<typename T, typename Operator>
__global__ void forAllKernel(T* array, int size, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < size; i += stride) {
        auto value = array[i];
        op(value);
    }
}

template<typename Operator>
__global__ void forAllnumVKernel(vid_t d_nV, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (vid_t i = id; i < d_nV; i += stride)
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

    for (vid_t i = id; i < num_items; i += stride)
        op(hornet.vertex(vertices_array[i]));
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

template<typename Operator>
void forAll(size_t size, const Operator& op) {
    detail::forAllKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (size, op);
}

template<typename T, typename Operator>
void forAll(const TwoLevelQueue<T>& queue, const Operator& op) {
    auto size = queue.size();
    detail::forAllKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (queue.device_input_ptr(), size, op);
}

//------------------------------------------------------------------------------

template<typename HornetClass, typename Operator>
void forAllnumV(HornetClass& hornet, const Operator& op) {
    detail::forAllnumVKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(hornet.nV()), BLOCK_SIZE_OP2 >>>
        (hornet.nV(), op);
}

//------------------------------------------------------------------------------

template<typename HornetClass, typename Operator>
void forAllnumE(HornetClass& hornet, const Operator& op) {
    detail::forAllnumEKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(hornet.nE()), BLOCK_SIZE_OP2 >>>
        (hornet.nE(), op);
}

//==============================================================================

template<typename HornetClass, typename Operator>
void forAllVertices(HornetClass& hornet, const Operator& op) {
    detail::forAllVerticesKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(hornet.nV()), BLOCK_SIZE_OP2 >>>
        (hornet.device_side(), op);
}

//------------------------------------------------------------------------------

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&         hornet,
                 const Operator&      op,
                 const LoadBalancing& load_balacing) {
    const int PARTITION_SIZE = xlib::SMemPerBlock<BLOCK_SIZE_OP2, vid_t>::value;
    int num_partitions = xlib::ceil_div<PARTITION_SIZE>(hornet.nE());

    load_balacing.apply(hornet, op);
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
}

template<typename HornetClass, typename Operator>
void forAllVertices(HornetClass&                hornet,
                    const TwoLevelQueue<vid_t>& queue,
                    const Operator&             op) {
    unsigned size = queue.size();
    detail::forAllVerticesKernel
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(size), BLOCK_SIZE_OP2 >>>
        (hornet.device_side(), queue.device_input_ptr(), size, op);
}

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&    hornet,
                 const vid_t*    vertex_array,
                 int             size,
                 const Operator& op,
                 const LoadBalancing& load_balacing) {
    load_balacing.apply(hornet, vertex_array, size, op);
}
/*
template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass& hornet,
                 const TwoLevelQueue<vid_t>& queue,
                 const Operator& op, const LoadBalancing& load_balacing) {
    load_balacing.apply(hornet, queue.device_input_ptr(),
                        queue.size(), op);
    //queue.kernel_after();
}*/

template<typename HornetClass, typename Operator, typename LoadBalancing>
void forAllEdges(HornetClass&                hornet,
                 const TwoLevelQueue<vid_t>& queue,
                 const Operator&             op,
                 const LoadBalancing&        load_balacing) {    
    load_balacing.apply(hornet, queue.device_input_ptr(), queue.size(), op);
}

} // namespace hornet_alg
