
template<unsigned BLOCK_SIZE, typename VType, unsigned UNROLL_STEPS = 1>
class Tile {
    static const unsigned        RATIO = sizeof(VType) / sizeof(T);
public:

    __device__ __forceinline__
    bool is_valid() const {
        return _index < _size;
    }

    __device__ __forceinline__
    int last_index() const {
        return _size * THREAD_ITEMS;
    }

private:
    int _index;
    int _size;
    T*  _ptr;
    int _stride;
    int _full_stride
};


    template<typename T, int SIZE>
    __device__ __forceinline__
    void write(const T (&array)[SIZE]) {

    }


template<unsigned BLOCK_SIZE, typename VType, unsigned UNROLL_STEPS = 1>
class IlLoadTile : public Tile<> {
    static const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;

    template<typename T>
    IlLoad(const T* ptr, int num_items) :
                _index(blockIdx.x * BLOCK_SIZE + threadIdx.x),
                _size(xlib::lower_approx<WARP_SIZE>(num_items / THREAD_ITEMS)),
                _stride(gridDim.x * BLOCK_SIZE),
                _ptr(ptr + _index * RATIO),
                _full_stride(_stride * THREAD_ITEMS) {

        static_assert(sizeof(VType) % sizeof(T) == 0,
                      "VType and T do not match");
        assert(xlib::is_aligned<VType>(ptr) && "ptr not aligned to VType");
    }

    template<typename T, int SIZE>
    void load(T (&array)[SIZE]) {
        const auto& d_in = reinterpret_cast<const VType*>(_ptr);
        auto&      l_out = reinterpret_cast<VType*>(array);

        #pragma unroll
        for (int J = 0; J < UNROLL_STEPS; J++) {
            l_out[J * 2]     = d_in[_stride * J * 2];
            l_out[J * 2 + 1] = __ldg(&d_in[_stride * (J * 2 + 1)]);
        }
        _index += _full_stride;
        _ptr   += _full_stride;
    }

    template<typename T, int SIZE>
    __device__ __forceinline__
    void load(T (&array)[SIZE], int (&indices)[SIZE]) {
        const auto& d_in = reinterpret_cast<const VType*>(_ptr);
        auto&      l_out = reinterpret_cast<VType*>(array);

        #pragma unroll
        for (int J = 0; J < UNROLL_STEPS; J++) {
            #pragma unroll
            for (int K = 0; K < RATIO; K++) {
                const int STEP1 = RATIO * (J * 2) + K;
                const int STEP2 = RATIO * (J * 2 + 1) + K;
                indices[STEP1]  = RATIO * (i + _stride * J * 2) + K;
                indices[STEP2]  = RATIO * (i + _stride * (J * 2 + 1)) + K;
            }
            l_out[J * 2]     = d_in[_stride * J * 2];
            l_out[J * 2 + 1] = __ldg(&d_in[_stride * (J * 2 + 1)]);
        }
        _index += _full_stride;
        _ptr   += _full_stride;
    }
};

template<unsigned BLOCK_SIZE, typename VType, unsigned UNROLL_STEPS = 1>
class LoadTile : public Tile<> {
public:

    template<typename T, int SIZE>
    void load(T (&array)[SIZE]) {
        const auto& d_in = reinterpret_cast<const VType*>(_ptr);
        auto&      l_out = reinterpret_cast<VType*>(array);

        #pragma unroll
        for (int J = 0; J < UNROLL_STEPS; J++)
            l_out[J] = d_in[stride * J];
        _index += _full_stride;
        _ptr   += _full_stride;
    }

    template<typename T, int SIZE>
    __device__ __forceinline__
    void load(T (&array)[SIZE], int (&indices)[SIZE]) {
        const auto& d_in = reinterpret_cast<const VType*>(_ptr);
        auto&      l_out = reinterpret_cast<VType*>(array);

        #pragma unroll
        for (int J = 0; J < UNROLL_STEPS; J++) {
            #pragma unroll
            for (int K = 0; K < RATIO; K++)
                indices[RATIO * J + K] = RATIO * (_index + _stride * J) + K;
            l_out[J] = d_in[_stride * J];
        }
        _index += _full_stride;
        _ptr   += _full_stride;
    }
