
namespace custinger_alg {

inline StaticAlgorithm::StaticAlgorithm(custinger::cuStinger& custinger_)
                                        noexcept :
                                    custinger(custinger_),
                                    load_balacing(custinger_) {}

inline StaticAlgorithm::~StaticAlgorithm() noexcept {
    cuFree(_d_ptr);
}

template<typename T>
inline T* StaticAlgorithm::register_data(T& data) noexcept {
    if (_is_registered)
        ERROR("register_data() must be called only one times")
    _is_registered = true;
    _data_size     = sizeof(T);
    _h_ptr         = &data;
    SAFE_CALL( cudaMalloc(&_d_ptr, _data_size) )
    return reinterpret_cast<T*>(_d_ptr);
}

inline void StaticAlgorithm::syncHostWithDevice() noexcept {
    if (!_is_registered)
        ERROR("register_data() must be called before syncHostWithDevice()")
    SAFE_CALL( cudaMemcpy(_h_ptr, _d_ptr, _data_size, cudaMemcpyDeviceToHost) )
}

inline void StaticAlgorithm::syncDeviceWithHost() noexcept {
    if (!_is_registered)
        ERROR("register_data() must be called before syncDeviceWithHost()")
    SAFE_CALL( cudaMemcpy(_d_ptr, _h_ptr, _data_size, cudaMemcpyHostToDevice) )
}

//==============================================================================

template<typename T>
inline Allocate::Allocate(T*& pointer, size_t num_items) noexcept {
    cuMalloc(pointer, num_items);
    _pointer = pointer;
}

inline Allocate::~Allocate() noexcept {
    cuFree(_pointer);
}

} // namespace custinger_alg
