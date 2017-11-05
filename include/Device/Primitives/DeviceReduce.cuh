/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

namespace xlib {
namespace device_reduce {

const unsigned BLOCK_SIZE = 32;

namespace kernel {

template<unsigned UNROLL_STEPS = 1, typename R, typename T,
         typename ThreadOp, typename WarpOp, typename SeletOp>
__global__
void reduce_arg(T*                  __restrict__ d_in,
                int                              num_items,
                const ThreadOp&                  thread_op,
                const WarpOp&                    warp_op,
                const SeletOp&                   select_op,
                long long unsigned* __restrict__ d_out,
                R                                init_value) {

    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;

    int         idx = blockIdx.x * blockDim.x + threadIdx.x;
    int approx_size = xlib::lower_approx<WARP_SIZE>(num_items / THREAD_ITEMS);
    int      stride = gridDim.x * blockDim.x;

    auto d_tmp = d_in + idx * RATIO;
    for (int i = idx; i < approx_size; i += stride * THREAD_ITEMS) {
        T   storage[THREAD_ITEMS];
        int indices[THREAD_ITEMS];
        #pragma unroll
        for (int J = 0; J < UNROLL_STEPS; J++) {
            #pragma unroll
            for (int K = 0; K < RATIO; K++) {
                indices[RATIO * (J * 2) + K]     = RATIO * (i + stride * J * 2) + K;
                indices[RATIO * (J * 2 + 1) + K] = RATIO * (i + stride * (J * 2 + 1)) + K;
            }

            reinterpret_cast<int4*>(storage)[J * 2] =
                                 reinterpret_cast<int4*>(d_tmp)[stride * J * 2];
            reinterpret_cast<int4*>(storage)[J * 2 + 1] =
                   __ldg(&reinterpret_cast<int4*>(d_tmp)[stride * (J * 2 + 1)]);
        }
        R array[THREAD_ITEMS];
        #pragma unroll
        for (int J = 0; J < THREAD_ITEMS; J++)
            array[J] = select_op(storage[J]);

        #pragma unroll
        for (int STRIDE = 1; STRIDE < THREAD_ITEMS; STRIDE *= 2) {
            #pragma unroll
            for (int INDEX = 0; INDEX < THREAD_ITEMS; INDEX += STRIDE * 2) {
                thread_op(array[INDEX], indices[INDEX],
                          array[INDEX + STRIDE], indices[INDEX + STRIDE]);
            }
        }
        warp_op(array[0], indices[0], d_out);
        d_tmp += stride * THREAD_ITEMS;
    }
    R reduction = init_value;
    int index   = approx_size * THREAD_ITEMS + idx;
    stride      = blockDim.x * gridDim.x;
    if (xlib::lower_approx<WARP_SIZE>(index) >= num_items)
        return;
    for (int i = index; i < num_items; i += stride)
        thread_op(reduction, index, select_op(d_in[i]), i);
    warp_op(reduction, index, d_out);
}

} // namespace kernel

template<unsigned UNROLL_STEPS = 1, typename R, typename T, typename SeletOp,
         typename ThreadOp, typename AtomicOp>
typename std::pair<R, int>
reduce_arg(const T* __restrict__ d_in, int num_items, const SeletOp& select_op,
           const ThreadOp& thread_op, const AtomicOp& atomic_op,
           long long unsigned init_value) {

    using ULL = long long unsigned;
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;

    const auto& warp_op = [=] __device__ (R& value, int index,
                                          ULL* __restrict__ d_out) {
                #pragma unroll
                for (int i = WARP_SIZE / 2; i >= 1; i /= 2) {
                    auto tmp_value = xlib::shfl_xor(value, i);
                    int  tmp_index = __shfl_xor(index, i);
                    thread_op(value, index, tmp_value, tmp_index);
                }
        //printf("-->%d %d\t %d \t %d\n", blockIdx.x, threadIdx.x, value, index);

                int  value_int = reinterpret_cast<int&>(value);
                auto    value2 = make_int2(value_int, index);
                auto value_ull = reinterpret_cast<long long unsigned&>(value2);
                atomic_op(d_out, value_ull);
            };
    ULL *d_out;
    int2 h_out;
    SAFE_CALL( cudaMalloc(&d_out, sizeof(ULL)) )
    SAFE_CALL( cudaMemcpy(d_out, &init_value, sizeof(ULL),
                          cudaMemcpyHostToDevice))

    device_reduce::kernel::reduce_arg<UNROLL_STEPS, R>
        <<< xlib::ceil_div<BLOCK_SIZE * THREAD_ITEMS>(num_items), BLOCK_SIZE >>>
       (const_cast<T*>(d_in), num_items, thread_op, warp_op, select_op, d_out,
        reinterpret_cast<const R&>(init_value));

    SAFE_CALL( cudaMemcpy(&h_out, d_out, sizeof(ULL), cudaMemcpyDeviceToHost) )
    SAFE_CALL( cudaFree(d_out) )
    return std::pair<R, int>(reinterpret_cast<R&>(h_out.x), h_out.y);
}

template<unsigned UNROLL_STEPS = 1, typename R, typename T, typename SeletOp>
typename std::pair<R, int>
reduce_argmax(const T* __restrict__ d_in, int num_items,
              const SeletOp& select_op) {

    const auto   init_ull = std::numeric_limits<long long unsigned>::lowest();
    const auto& thread_op = [] __device__ (R& valueA, int& indexA,
                                           const R& valueB, int indexB) {
                                    if (valueB > valueA) {
                                       valueA = valueB;
                                       indexA = indexB;
                                    }
                                };
    const auto& atomic_op = [] __device__ (long long unsigned* d_out,
                                           long long unsigned value_ull) {
                                atomicMax(d_out, value_ull);
                            };
    return reduce_arg<UNROLL_STEPS, R>(d_in, num_items, select_op, thread_op,
                                       atomic_op, init_ull);
}

template<unsigned UNROLL_STEPS = 1, typename T>
typename std::pair<T, int>
argMax(const T* __restrict__ d_in, int num_items) {
    const auto& select_op = [] __device__ (const T& value) { return value; };
    return device_reduce::reduce_argmax<UNROLL_STEPS, T>
            (d_in, num_items, select_op);
}

//==============================================================================
//==============================================================================

namespace kernel {

template<unsigned UNROLL_STEPS = 1, typename T, typename R,
         typename ThreadOp, typename WarpOp>
__device__ __forceinline__
void reduce(T* __restrict__ d_in,
            int             num_items,
            const ThreadOp& thread_op,
            const WarpOp&   warp_op,
            R* __restrict__ d_out,
            const T&        zero_value) {

    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    //const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;
    const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO;

    int         idx = blockIdx.x * blockDim.x + threadIdx.x;
    int      stride = blockDim.x * gridDim.x;
    int approx_size = xlib::lower_approx<WARP_SIZE>(num_items / THREAD_ITEMS);

    d_in += idx * RATIO;
    for (int i = idx; i < approx_size; i += stride * THREAD_ITEMS) {
        T array[THREAD_ITEMS];
        #pragma unroll
        for (int J = 0; J < UNROLL_STEPS; J++) {
            reinterpret_cast<int4*>(array)[J] =
                                  reinterpret_cast<int4*>(d_in)[stride * J];
            //reinterpret_cast<int4*>(array)[J * 2] =
            //                      reinterpret_cast<int4*>(d_in)[stride * J * 2];
            //reinterpret_cast<int4*>(array)[J * 2 + 1] =
            //        __ldg(&reinterpret_cast<int4*>(d_in)[stride * (J * 2 + 1)]);
        }
        #pragma unroll
        for (int STRIDE = 1; STRIDE < THREAD_ITEMS; STRIDE *= 2) {
            #pragma unroll
            for (int INDEX = 0; INDEX < THREAD_ITEMS; INDEX += STRIDE * 2)
                array[INDEX] = thread_op(array[INDEX], array[INDEX + STRIDE]);
        }
        warp_op(array[0], d_out);
        d_in += stride * THREAD_ITEMS;
    }
    T reduction = zero_value;
    for(int i = approx_size * THREAD_ITEMS + idx; i < num_items; i += stride) {
       reduction = thread_op(reduction, *d_in);
       d_in += stride;
    }
    warp_op(reduction, d_out);
}

template<unsigned UNROLL_STEPS = 1, typename T, typename R,
         typename ThreadOp, typename WarpOp>
__global__
void reduceGlobal(T* __restrict__ d_in,
                  int             num_items,
                  ThreadOp        thread_op,
                  WarpOp          warp_op,
                  R* __restrict__ d_out,
                  T               zero_value) {
    reduce<UNROLL_STEPS>
        (d_in, num_items, thread_op, warp_op, d_out, zero_value);
}

//------------------------------------------------------------------------------

template<unsigned UNROLL_STEPS, typename T, typename R>
__global__
void add(T* __restrict__ d_in, int num_items, R* __restrict__ d_out,
         T zero_value) {

    const auto& thread_op = [] (const T& a, const T& b) { return a + b; };
    const auto&   warp_op = [] (const T& value, R* __restrict__ d_out) {
                                    WarpReduce<>::atomicAdd(value, d_out);
                                };
    reduce<UNROLL_STEPS>
        (d_in, num_items, thread_op, warp_op, d_out, zero_value);
}

template<unsigned UNROLL_STEPS, typename T, typename R>
__global__
void min(T* __restrict__ d_in, int num_items, R* __restrict__ d_out,
         T zero_value) {

    const auto& thread_op = [] (const T& a, const T& b) { return ::min(a, b); };
    const auto&   warp_op = [] (const T& value, R* __restrict__ d_out) {
                                    WarpReduce<>::atomicMin(value, d_out);
                                };
    reduce<UNROLL_STEPS>
        (d_in, num_items, thread_op, warp_op, d_out, zero_value);
}

template<unsigned UNROLL_STEPS, typename T, typename R>
__global__
void max(T* __restrict__ d_in, int num_items, R* __restrict__ d_out,
         T zero_value) {

    const auto& thread_op = [] (const T& a, const T& b) { return ::max(a, b); };
    const auto&   warp_op = [] (const T& value, R* __restrict__ d_out) {
                                    WarpReduce<>::atomicMax(value, d_out);
                                };
    reduce<UNROLL_STEPS>
        (d_in, num_items, thread_op, warp_op, d_out, zero_value);
}

} // namespace kernel

//==============================================================================

template<unsigned UNROLL_STEPS = 1, typename T, typename R,
         typename ThreadOp, typename WarpOp>
R apply(const T* __restrict__ d_in,
         int                  num_items,
         const ThreadOp&      thread_op,
         const WarpOp&        warp_op,
         const R&             init_value,
         const T&             zero_value = T()) {
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;

    R h_out, *d_out;
    SAFE_CALL( cudaMalloc(&d_out, sizeof(R)) )
    SAFE_CALL(cudaMemcpy(d_out, &init_value, sizeof(R), cudaMemcpyHostToDevice))

    device_reduce::kernel::reduceGlobal<UNROLL_STEPS>
        <<< xlib::ceil_div<BLOCK_SIZE * THREAD_ITEMS>(num_items), BLOCK_SIZE >>>
       (const_cast<T*>(d_in), num_items, thread_op, warp_op, d_out, zero_value);

    SAFE_CALL( cudaMemcpy(&h_out, d_out, sizeof(R), cudaMemcpyDeviceToHost) )
    SAFE_CALL( cudaFree(d_out) )
    return h_out;
}

//------------------------------------------------------------------------------

template<unsigned UNROLL_STEPS = 1, typename T, typename R = T>
T max(const T* __restrict__ d_in, int num_items,
      const R& init_value = std::numeric_limits<R>::lowest(),
      const T& zero_value = T()) {
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;

    R h_out, *d_out;
    SAFE_CALL( cudaMalloc(&d_out, sizeof(R)) )
    SAFE_CALL(cudaMemcpy(d_out, &init_value, sizeof(R), cudaMemcpyHostToDevice))

    device_reduce::kernel::max<UNROLL_STEPS>
        <<< xlib::ceil_div<BLOCK_SIZE * THREAD_ITEMS>(num_items), BLOCK_SIZE >>>
        (const_cast<T*>(d_in), num_items, d_out, zero_value);

    SAFE_CALL( cudaMemcpy(&h_out, d_out, sizeof(R), cudaMemcpyDeviceToHost) )
    SAFE_CALL( cudaFree(d_out) )
    return h_out;
}

template<unsigned UNROLL_STEPS = 1, typename T, typename R>
void max(const T* __restrict__ d_in, int num_items, R* __restrict__ d_out,
         const T& zero_value = T()) {
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;

    device_reduce::kernel::max<UNROLL_STEPS>
        <<< xlib::ceil_div<BLOCK_SIZE * THREAD_ITEMS>(num_items), BLOCK_SIZE >>>
        (const_cast<T*>(d_in), num_items, d_out, zero_value);
}

//------------------------------------------------------------------------------

template<unsigned UNROLL_STEPS = 1, typename T, typename R = T>
T add(const T* __restrict__ d_in, int num_items, const T& zero_value = T()) {
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    //const unsigned THREAD_ITEMS = UNROLL_STEPS * 2 * RATIO;
    const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO;

    R h_out, *d_out;
    SAFE_CALL( cudaMalloc(&d_out, sizeof(R)) )
    const auto value = R(0);
    SAFE_CALL( cudaMemcpy(d_out, &value, sizeof(R), cudaMemcpyHostToDevice) )

    device_reduce::kernel::add<UNROLL_STEPS>
        <<< xlib::ceil_div<BLOCK_SIZE * THREAD_ITEMS>(num_items), BLOCK_SIZE >>>
        (const_cast<T*>(d_in), num_items, d_out, zero_value);
    CHECK_CUDA_ERROR

    SAFE_CALL( cudaMemcpy(&h_out, d_out, sizeof(R), cudaMemcpyDeviceToHost) )
    SAFE_CALL( cudaFree(d_out) )
    return h_out;
}

template<unsigned UNROLL_STEPS = 1, typename T, typename R>
void add(const T* __restrict__ d_in, int num_items, R* __restrict__ d_out,
         const T& zero_value = T()) {
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO;

    device_reduce::kernel::add<UNROLL_STEPS>
        <<< xlib::ceil_div<BLOCK_SIZE * THREAD_ITEMS>(num_items), BLOCK_SIZE >>>
        (const_cast<T*>(d_in), num_items, d_out, zero_value);
}

} // namespace device_reduce
} // namespace xlib
