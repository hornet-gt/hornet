/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
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
#include "Device/Primitives/CubWrapper.cuh"
#include "Device/Util/SafeCudaAPI.cuh"
#include "Device/Util/SafeCudaAPISync.cuh"
#include "Device/Util/VectorUtil.cuh"
#include "Host/Numeric.hpp"

#if defined(CUB_WRAPPER)

#include <cub/cub.cuh>

namespace xlib {

CubWrapper::CubWrapper(const int num_items) noexcept : _num_items(num_items) {}

void CubWrapper::initialize(const int num_items) noexcept {
    _num_items = num_items;
}

void CubWrapper::release(void) noexcept {
    cuFree(_d_temp_storage);
    _num_items = 0;
}

CubWrapper::~CubWrapper() noexcept {
    release();
}

//==============================================================================
//==============================================================================

template<typename T>
CubReduce<T>::CubReduce(const T* d_in, size_t num_items) noexcept :
                            CubWrapper(num_items), _d_in(d_in) {
    cuMalloc(_d_out, 1);
    cub::DeviceReduce::Sum(_d_temp_storage, _temp_storage_bytes,
                           _d_in, _d_out, _num_items);
    cuMalloc(_d_temp_storage, _temp_storage_bytes);
}

template<typename T>
T CubReduce<T>::run() noexcept {
    cub::DeviceReduce::Sum(_d_temp_storage, _temp_storage_bytes,
                           _d_in, _d_out, _num_items);
    int h_result;
    cuMemcpyToHost(_d_out, h_result);
    return h_result;
}

template<typename T>
CubReduce<T>::~CubReduce() noexcept {
    cuFree(_d_out);
}
//------------------------------------------------------------------------------

template<typename T>
CubSegmentedReduce<T>::CubSegmentedReduce(int* d_offsets, const T* d_in,
                                          int num_segments, T*& d_out) :
                                   CubWrapper(num_segments), _d_in(d_in),
                                   _d_out(d_out), _d_offsets(d_offsets) {

    cub::DeviceSegmentedReduce::Sum(_d_temp_storage, _temp_storage_bytes,
                                    _d_in, _d_out, num_segments,
                                    _d_offsets, _d_offsets + 1);
    cuMalloc(_d_temp_storage, _temp_storage_bytes);
    cuMalloc(d_out, num_segments);
}

template<typename T>
CubSegmentedReduce<T>::~CubSegmentedReduce() noexcept {
    cuFree(_d_out);
}

template<typename T>
void CubSegmentedReduce<T>::run() noexcept {
    cub::DeviceSegmentedReduce::Sum(_d_temp_storage, _temp_storage_bytes,
                                    _d_in, _d_out, _num_items,
                                    _d_offsets, _d_offsets + 1);
}

//------------------------------------------------------------------------------

template<typename T>
CubSpMV<T>::CubSpMV(T* d_value, int* d_row_offsets, int* d_column_indices,
                    T* d_vector_x, T* d_vector_y,
                    int num_rows, int num_cols, int num_nonzeros) :
                       CubWrapper(0),
                       _d_row_offsets(d_row_offsets),
                       _d_column_indices(d_column_indices),
                       _d_values(d_value),
                       _d_vector_x(d_vector_x), _d_vector_y(d_vector_y),
                       _num_rows(num_rows), _num_cols(num_cols),
                       _num_nonzeros(num_nonzeros) {

    cub::DeviceSpmv::CsrMV(_d_temp_storage, _temp_storage_bytes,
                           _d_values, _d_row_offsets, _d_column_indices,
                           _d_vector_x, _d_vector_y,
                           _num_rows, _num_cols, _num_nonzeros);
    cuMalloc(_d_temp_storage, _temp_storage_bytes);
}
/*
template<typename T>
CubSpMV<T>::~CubSpMV() noexcept {
    cuFree(_d_out);
}*/

template<typename T>
void CubSpMV<T>::run() noexcept {
    cub::DeviceSpmv::CsrMV(_d_temp_storage, _temp_storage_bytes, _d_values,
                           _d_row_offsets, _d_column_indices,
                           _d_vector_x, _d_vector_y,
                           _num_rows, _num_cols, _num_nonzeros);
}

//------------------------------------------------------------------------------

template<typename T>
CubArgMax<T>::CubArgMax(const T* d_in, size_t num_items) noexcept :
                                    _d_in(d_in), CubWrapper(num_items) {
    cub::KeyValuePair<int, T>* d_tmp;
    cuMalloc(d_tmp, 1);
    cub::DeviceReduce::ArgMax(_d_temp_storage, _temp_storage_bytes, _d_in,
                              static_cast<cub::KeyValuePair<int, T>*>(_d_out),
                              _num_items);
    cuMalloc(_d_temp_storage, _temp_storage_bytes);
    _d_out = reinterpret_cast<cub::KeyValuePair<int, T>*>(d_tmp);
}

template<typename T>
typename std::pair<int, T>
CubArgMax<T>::run() noexcept {
    cub::DeviceReduce::ArgMax(_d_temp_storage, _temp_storage_bytes, _d_in,
                              static_cast<cub::KeyValuePair<int, T>*>(_d_out),
                              _num_items);
    cub::KeyValuePair<int, T> h_out;
    cuMemcpyToHost(static_cast<cub::KeyValuePair<int, T>*>(_d_out), h_out);
    return std::pair<int, T>(h_out.key, h_out.value);
}

//==============================================================================
//==============================================================================
/////////////////
// SortByValue //
/////////////////

template<typename T>
CubSortByValue<T>::CubSortByValue(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
void CubSortByValue<T>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    size_t temp_storage_bytes = 0;
    T* d_in = nullptr, *d_sorted = nullptr;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                                   d_in, d_sorted, _num_items,
                                   0, sizeof(T) * 8);
    cuMalloc(_d_temp_storage, temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubSortByValue<T>::run(const T* d_in, int num_items, T* d_sorted,
                            T d_value_max) noexcept {
    size_t temp_storage_bytes;
    int num_bits = std::is_floating_point<T>::value ? sizeof(T) * 8 :
                                                   xlib::ceil_log2(d_value_max);
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                                   d_in, d_sorted, num_items, 0, num_bits);

    cub::DeviceRadixSort::SortKeys(_d_temp_storage, temp_storage_bytes,
                                   d_in, d_sorted, num_items, 0, num_bits);
}

//------------------------------------------------------------------------------

template<typename T>
void CubSortByValue<T>::srun(const T* d_in, int num_items, T* d_sorted,
                             T d_in_max) noexcept {
    CubSortByValue<T> cub_instance(num_items);
    cub_instance.run(d_in, num_items, d_sorted, d_in_max);
}

//==============================================================================
//==============================================================================
///////////////
// SortByKey //
///////////////

template<typename T, typename R>
CubSortByKey<T, R>::CubSortByKey(const int max_items) noexcept  {
    initialize(max_items);
}

template<typename T, typename R>
void CubSortByKey<T, R>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    size_t temp_storage_bytes = 0;
    T* d_key = nullptr, *d_key_sorted = nullptr;
    R* d_data_in = nullptr, *d_data_out = nullptr;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    d_key, d_key_sorted,
                                    d_data_in, d_data_out,
                                    _num_items, 0, sizeof(T) * 8);
    cuMalloc(_d_temp_storage, temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortByKey<T, R>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        CubWrapper::release();
        initialize(max_items);
    }
}

template<typename T, typename R>
void CubSortByKey<T, R>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        CubWrapper::release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortByKey<T, R>::run(
        const T* d_key,
        const R* d_data_in,
        const int num_items,
        T* d_key_sorted,
        R* d_data_out,
        T d_key_max) noexcept {
    int temp_num_items = num_items;
    using U = typename std::conditional<std::is_floating_point<T>::value,
                                        int, T>::type;
    int num_bits = std::is_floating_point<T>::value ? sizeof(T) * 8 :
                                     xlib::ceil_log2(static_cast<U>(d_key_max));
    cub::DeviceRadixSort::SortPairs(nullptr, _temp_storage_bytes,
                                    d_key, d_key_sorted,
                                    d_data_in, d_data_out,
                                    temp_num_items, 0, num_bits);
    cub::DeviceRadixSort::SortPairs(_d_temp_storage, _temp_storage_bytes,
                                    d_key, d_key_sorted,
                                    d_data_in, d_data_out,
                                    temp_num_items, 0, num_bits);
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortByKey<T, R>::srun(
        const T* d_key, const R* d_data_in,
        const int num_items, T* d_key_sorted,
        R* d_data_out, T d_key_max) noexcept {
    CubSortByKey<T, R> cub_instance(num_items);
    cub_instance.run(d_key, d_data_in, num_items, d_key_sorted, d_data_out);
}

//------------------------------------------------------------------------------

namespace cub_sort_by_key {

template<typename T, typename R>
void run(const T* d_key,
         const R* d_data_in,
         const int      num_items,
         T*       d_key_sorted,
         R*       d_data_out,
         T        d_key_max) {

    CubSortByKey<T, R> cub_instance(num_items);
    cub_instance.run(d_key, d_data_in, num_items, d_key_sorted, d_data_out);
}

/*
template<typename T, typename R>
void run(const T* d_key,
         const R* d_data_in,
         int      num_items,
         T*       d_key_sorted,
         R*       d_data_out,
         T*       d_key_tmp,
         R*       d_data_tmp,
         T        d_key_max) noexcept {
}*/

template void run<int, int>(const int*, const int*, const int, int*, int*, int);
template void run<int, float>(const int*, const float*, const int, int*, float*, int);
template void run<int, double>(const int*, const double*, const int, int*, double*, int);

} // namespace cub_sort_by_key

//==============================================================================
//==============================================================================
////////////////
// SortPairs2 //
////////////////

template<typename T, typename R>
CubSortPairs2<T, R>::CubSortPairs2(const int max_items, const bool internal_allocation)
                                   noexcept {
    initialize(max_items, internal_allocation);
}

template<typename T, typename R>
CubSortPairs2<T, R>::~CubSortPairs2() noexcept {
    release();
}

template<typename T, typename R>
void CubSortPairs2<T, R>::initialize(
        const int max_items,
        const bool internal_allocation) noexcept {
    CubWrapper::initialize(max_items);
    _internal_alloc = internal_allocation;
    if (_internal_alloc) {
        cuMalloc(_d_in1_tmp, _num_items);
        cuMalloc(_d_in2_tmp, _num_items);
    }
    size_t temp_storage_bytes = 0;
    T* d_in1 = nullptr;
    R* d_in2 = nullptr;
    if (sizeof(T) > sizeof(R)) {
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        d_in1, _d_in1_tmp, d_in2, _d_in2_tmp,
                                        _num_items, 0, sizeof(T) * 8);
    }
    else {
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        d_in2, _d_in2_tmp, d_in1, _d_in1_tmp,
                                        _num_items, 0, sizeof(R) * 8);
    }
    cuMalloc(_d_temp_storage, temp_storage_bytes);
}

template<typename T, typename R>
void CubSortPairs2<T, R>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items, _internal_alloc);
    }
}

template<typename T, typename R>
void CubSortPairs2<T, R>::release(void) noexcept {
    if (_internal_alloc) {
        cuFree(_d_in1_tmp, _d_in2_tmp);
    }
    CubWrapper::release();
}

template<typename T, typename R>
void CubSortPairs2<T, R>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortPairs2<T, R>::run(T* d_in1, R* d_in2, const int num_items,
                              T* d_in1_tmp, R* d_in2_tmp,
                              T d_in1_max, R d_in2_max) noexcept {

    int temp_num_items = num_items;
    int num_bits1 = std::is_floating_point<T>::value ? sizeof(T) * 8 :
                                                   xlib::ceil_log2(d_in1_max);
    int num_bits2 = std::is_floating_point<R>::value ? sizeof(T) * 8 :
                                                   xlib::ceil_log2(d_in2_max);
    size_t temp_storage_bytes;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    d_in2, d_in2_tmp, d_in1, d_in1_tmp,
                                    temp_num_items, 0, num_bits2);
    cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes,
                                    d_in2, d_in2_tmp, d_in1, d_in1_tmp,
                                    temp_num_items, 0, num_bits2);

    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    d_in1_tmp, d_in1, d_in2_tmp, d_in2,
                                    temp_num_items, 0, num_bits1);
    cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes,
                                    d_in1_tmp, d_in1, d_in2_tmp, d_in2,
                                    temp_num_items, 0, num_bits1);
}

template<typename T, typename R>
void CubSortPairs2<T, R>::run(T* d_in1, R* d_in2, const int num_items,
                              T d_in1_max, R d_in2_max) noexcept {
    run(d_in1, d_in2, num_items, _d_in1_tmp, _d_in2_tmp, d_in1_max, d_in2_max);
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortPairs2<T, R>::srun(T* d_in1, R* d_in2, const int num_items,
                               T d_in1_max, R d_in2_max) noexcept {
    CubSortPairs2<T, R> cub_instance(num_items, true);
    cub_instance.run(d_in1, d_in2, num_items, d_in1_max, d_in2_max);
}

template<typename T, typename R>
void CubSortPairs2<T, R>::srun(T* d_in1, R* d_in2, const int num_items,
                               T* d_in1_tmp, R* d_in2_tmp,
                               T d_in1_max, R d_in2_max) noexcept {
    CubSortPairs2<T, R> cub_instance(num_items, false);
    cub_instance.run(d_in1, d_in2, num_items, d_in1_tmp, d_in2_tmp,
                     d_in1_max, d_in2_max);
}

//==============================================================================
//==============================================================================
/////////////////////
// RunLengthEncode //
/////////////////////

namespace cub_runlength {

template<typename T>
int run(const T* d_in, const int num_items, T* d_unique_out,
        int* d_counts_out) {

    CubRunLengthEncode<T> cub_instance(num_items);
    return cub_instance.run(d_in, num_items, d_unique_out, d_counts_out);
}

template int run<int>(const int*, const int, int*, int*);

} // namespace cub_runlength

//------------------------------------------------------------------------------

template<typename T>
CubRunLengthEncode<T>::CubRunLengthEncode(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
CubRunLengthEncode<T>::~CubRunLengthEncode() noexcept {
    release();
}

template<typename T>
void CubRunLengthEncode<T>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    cuMalloc(_d_num_runs_out, 1);
    T* d_in = nullptr, *d_unique_out = nullptr;
    int* d_counts_out = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes,
                                       d_in, d_unique_out, d_counts_out,
                                       _d_num_runs_out, _num_items);
    cuMalloc(_d_temp_storage, temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubRunLengthEncode<T>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items);
    }
}

template<typename T>
void CubRunLengthEncode<T>::release(void) noexcept {
    cuFree(_d_num_runs_out);
    CubWrapper::release();
}

template<typename T>
void CubRunLengthEncode<T>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T>
int CubRunLengthEncode<T>::run(const T* d_in, const int num_items,
                               T* d_unique_out, int* d_counts_out) noexcept {
    int temp_num_items = num_items;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes,
                                       d_in, d_unique_out, d_counts_out,
                                       _d_num_runs_out, temp_num_items);
    cub::DeviceRunLengthEncode::Encode(_d_temp_storage, temp_storage_bytes,
                                       d_in, d_unique_out, d_counts_out,
                                       _d_num_runs_out, temp_num_items);
    int h_num_runs_out;
    cuMemcpyToHost(_d_num_runs_out, h_num_runs_out);
    return h_num_runs_out;
}

//------------------------------------------------------------------------------

template<typename T>
int CubRunLengthEncode<T>::srun(const T* d_in, const int num_items, T* d_unique_out,
                                int* d_counts_out) noexcept {
    CubRunLengthEncode<T> cub_instance(num_items);
    return cub_instance.run(d_in, num_items, d_unique_out, d_counts_out);
}

//==============================================================================
//==============================================================================
//////////////////
// ExclusiveSum //
//////////////////

namespace cub_exclusive_sum {

template<typename T>
void run(const T* d_in, const int num_items, T* d_out) {
    CubExclusiveSum<T> cub_instance(num_items);
    cub_instance.run(d_in, num_items, d_out);
}

template<typename T>
void run(T* d_in_out, const int num_items) {
    run(d_in_out, num_items, d_in_out);
}

template void run<int>(const int*, const int, int*);
template void run<int>(int*, const int);

} // namespace cub_exclusive_sum

//------------------------------------------------------------------------------

template<typename T>
CubExclusiveSum<T>::CubExclusiveSum(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
void CubExclusiveSum<T>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    size_t temp_storage_bytes = 0;
    T* d_in = nullptr, *d_out = nullptr;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                  d_in, d_out, _num_items);
    cuMalloc(_d_temp_storage, temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubExclusiveSum<T>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items);
    }
}

template<typename T>
void CubExclusiveSum<T>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T>
void CubExclusiveSum<T>::run(
        const T* d_in,
        const int num_items,
        T* d_out) const noexcept {
    int temp_num_items = num_items;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                  d_in, d_out, temp_num_items);
    cub::DeviceScan::ExclusiveSum(_d_temp_storage, temp_storage_bytes,
                                  d_in, d_out, temp_num_items);
}

template<typename T>
void CubExclusiveSum<T>::run(T* d_in_out, const int num_items) const noexcept {
    run(d_in_out, num_items, d_in_out);
}

//------------------------------------------------------------------------------

template<typename T>
void CubExclusiveSum<T>::srun(const T* d_in, const int num_items, T* d_out) noexcept {
    CubExclusiveSum<T> cub_instance(num_items);
    cub_instance.run(d_in, num_items, d_out);
}

template<typename T>
void CubExclusiveSum<T>::srun(T* d_in_out, const int num_items) noexcept {
    CubExclusiveSum::srun(d_in_out, num_items, d_in_out);
}

//==============================================================================
//==============================================================================
/////////////////////
// CubInclusiveMax //
/////////////////////

// CubMax functor
struct CubMax
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

template<typename T>
CubInclusiveMax<T>::CubInclusiveMax(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
void CubInclusiveMax<T>::initialize(const int max_items) noexcept {
    CubMax max_op;
    CubWrapper::initialize(max_items);
    size_t temp_storage_bytes;
    T* d_in = nullptr, *d_out = nullptr;
    cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes,
                                  d_in, d_out, max_op, _num_items);
    cuMalloc(_d_temp_storage, _temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubInclusiveMax<T>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items);
    }
}

template<typename T>
void CubInclusiveMax<T>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T>
void CubInclusiveMax<T>::run(
        const T* d_in,
        const int num_items,
        T* d_out) const noexcept {
    CubMax max_op;
    int temp_num_items = num_items;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes,
                                  d_in, d_out, max_op, temp_num_items);
    cub::DeviceScan::InclusiveScan(_d_temp_storage, temp_storage_bytes,
                                  d_in, d_out, max_op, temp_num_items);
}

template<typename T>
void CubInclusiveMax<T>::run(T* d_in_out, const int num_items) const noexcept {
    run(d_in_out, num_items, d_in_out);
}

//------------------------------------------------------------------------------

template<typename T>
void CubInclusiveMax<T>::srun(const T* d_in, const int num_items, T* d_out) noexcept {
    CubInclusiveMax<T> cub_instance(num_items);
    cub_instance.run(d_in, num_items, d_out);
}

template<typename T>
void CubInclusiveMax<T>::srun(T* d_in_out, const int num_items) noexcept {
    CubInclusiveMax::srun(d_in_out, num_items, d_in_out);
}



//==============================================================================
//==============================================================================
///////////////////
// SelectFlagged //
///////////////////

template<typename T>
CubSelectFlagged<T>::CubSelectFlagged(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
CubSelectFlagged<T>::~CubSelectFlagged() noexcept {
    release();
}

template<typename T>
void CubSelectFlagged<T>::release(void) noexcept {
    cuFree(_d_num_selected_out);
    CubWrapper::release();
}

template<typename T>
void CubSelectFlagged<T>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    cuMalloc(_d_num_selected_out, 1);
    size_t temp_storage_bytes = 0;
    T* d_in = nullptr, *d_out = nullptr;
    bool* d_flags = nullptr;

    cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, d_in,
                               d_flags, d_out, _d_num_selected_out,
                               _num_items);
    cuMalloc(_d_temp_storage, temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubSelectFlagged<T>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items);
    }
}

template<typename T>
void CubSelectFlagged<T>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T>
int CubSelectFlagged<T>::run(const T* d_in, const int num_items,
                             const bool* d_flags, T* d_out) noexcept {
    int temp_num_items = num_items;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, d_in,
                               d_flags, d_out, _d_num_selected_out,
                               temp_num_items);
    cub::DeviceSelect::Flagged(_d_temp_storage, temp_storage_bytes, d_in,
                               d_flags, d_out, _d_num_selected_out,
                               temp_num_items);
    int h_num_selected_out;
    cuMemcpyToHost(_d_num_selected_out, h_num_selected_out);
    return h_num_selected_out;
}

template<typename T>
int CubSelectFlagged<T>::run(T* d_in_out, const int num_items, const bool* d_flags)
                             noexcept {
    return run(d_in_out, num_items, d_flags, d_in_out);
}

//------------------------------------------------------------------------------

template<typename T>
int CubSelectFlagged<T>::srun(const T* d_in, const int num_items, const bool* d_flags,
                              T* d_out) noexcept {
    CubSelectFlagged cub_instance(num_items);
    return cub_instance.run(d_in, num_items, d_flags, d_out);
}

template<typename T>
int CubSelectFlagged<T>::srun(T* d_in_out, const int num_items, const bool* d_flags)
                              noexcept {
    return CubSelectFlagged::srun(d_in_out, num_items, d_flags, d_in_out);
};

//==============================================================================
//==============================================================================

template class CubArgMax<int>;
template class CubSortByValue<int>;
template class CubSortByKey<int, int>;
template class CubSortByKey<double, int>;
template class CubSortPairs2<int, int>;
template class CubRunLengthEncode<int>;
template class CubExclusiveSum<int>;
template class CubInclusiveMax<int>;
template class CubSelectFlagged<int>;

//template class CubSelectFlagged<hornets_nest::AoSData<int>>;

} // namespace xlib

#endif

