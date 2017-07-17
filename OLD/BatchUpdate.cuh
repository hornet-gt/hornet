/**
 * @brief Batch Property
 */
class BatchInit {
public:
    /**
     * @brief default costructor
     * @param[in] sort the edge batch is sorted in lexicographic order
     *            (source, destination)
     * @param[in] weighted_distr generate a batch by using a random weighted
     *            distribution based on the degree of the vertices
     * @param[in] print print the batch on the standard output
     */
    explicit BatchInit(const vid_t* src_array,  const vid_t* dst_array,
                       int batch_size) noexcept;

    /**
     * @brief Insert additional edge data
     * @param[in] edge_data list of edge data. The list must contains atleast
     *            the source and the destination arrays (vid_t type)
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *config.inc* file
     * @see ::insertVertexData
     */
    template<typename... TArgs>
    void insertEdgeData(TArgs... edge_data) noexcept;

    int size() const noexcept;

    const byte_t* data_ptrs(int index) const noexcept;

private:
    const byte_t* _edge_ptrs[ NUM_ETYPES + 1 ] = {};
    const int     _batch_size { 0 };
};


/**
 * @brief Batch update class
 */
class BatchUpdate {
    friend class cuStinger;
public:
    /**
     * @brief default costructor
     * @param[in] batch_size number of edges of the batch
     */
    explicit BatchUpdate(const BatchInit& batch_init) noexcept;

    explicit BatchUpdate(size_t size) noexcept;

    //copy costructor to copy the batch to the kernel
    BatchUpdate(const BatchUpdate& obj) noexcept;

    ~BatchUpdate() noexcept;

    void sendToDevice(const BatchInit& batch_init) noexcept;

    void bind(const BatchInit& batch_init) noexcept;

#if defined(__NVCC__)

    __host__ __device__ __forceinline__
    int size() const noexcept;

    __device__ __forceinline__
    vid_t src(int index) const noexcept;

    __device__ __forceinline__
    vid_t dst(int index) const noexcept;

    __device__ __forceinline__
    Edge edge(int index) const noexcept;

    template<int INDEX>
    __device__ __forceinline__
    typename std::tuple_element<INDEX, VertexTypes>::type
    field(int index) const noexcept;

    __device__ __forceinline__
    eoff_t* offsets_ptr() const noexcept;

    __device__ __forceinline__
    int offsets_size() const noexcept;

    __host__ __device__ __forceinline__
    vid_t* src_ptr() const noexcept;

    __host__ __device__ __forceinline__
    vid_t* dst_ptr() const noexcept;
#endif

private:
    byte_t*    _pinned_ptr    { nullptr };
    byte_t*    _d_edge_ptrs[ NUM_ETYPES + 1 ] = {};
    eoff_t*    _d_offsets     { nullptr };
    int        _batch_size    { 0 };
    int        _offsets_size  { 0 };
    const int  _batch_pitch   { 0 }; //number of edges to the next field
    const bool _enable_delete { true };
};
