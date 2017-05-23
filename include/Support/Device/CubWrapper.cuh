/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

CudaStinger is provided under the terms of The MIT License (MIT):

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

#include <limits>

namespace xlib {

class CubWrapper {
protected:
    explicit CubWrapper(size_t num_items) noexcept;
    ~CubWrapper() noexcept;

    void*        _d_temp_storage     { nullptr };
    size_t       _temp_storage_bytes { 0 };
    const size_t _num_items;
};

template<typename T>
class CubSortByValue : public CubWrapper {
public:
    explicit CubSortByValue(const T* d_in, size_t size, T* d_sorted,
                            T d_in_max = std::numeric_limits<T>::max())
                            noexcept;
    void run() noexcept;

private:
    const T* _d_in;
    T*       _d_sorted;
    T        _d_in_max;
};

template<typename T, typename R>
class CubSortByKey : public CubWrapper {
public:
    CubSortByKey(const T* d_key, const R* d_data_in, size_t num_items,
                 T* d_key_sorted, R* d_data_out,
                 T d_key_max = std::numeric_limits<T>::max()) noexcept;

    void run() noexcept;

private:
    const T* _d_key;
    const R* _d_data_in;
    T*       _d_key_sorted;
    R*       _d_data_out;
    T        _d_key_max;
};

template<typename T, typename R>
class CubSortPairs2 : public CubWrapper {
public:
    CubSortPairs2(T* d_in1, R* d_in2, size_t num_items,
                  T d_in1_max = std::numeric_limits<T>::max(),
                  R d_in2_max = std::numeric_limits<R>::max());

    CubSortPairs2(const T* d_in1, const R* d_in2, size_t num_items,
                  T* d_out1, R* d_out2,
                  T d_in1_max = std::numeric_limits<T>::max(),
                  R d_in2_max = std::numeric_limits<R>::max());

    ~CubSortPairs2() noexcept;

    void run() noexcept;

private:
    T*       _d_in1, *_d_out1, *_d_in1_tmp { nullptr };
    R*       _d_in2, *_d_out2, *_d_in2_tmp { nullptr };
    T        _d_in1_max;
    R        _d_in2_max;
    bool     _internal_alloc { false };
};

template<typename T>
class CubUnique : public CubWrapper {
public:
    CubUnique(const T* d_in, size_t size, T*& d_unique_batch);
    ~CubUnique() noexcept;
    int run() noexcept;
private:
    const T* _d_in;
    T*&      _d_unique_batch;
    int*     _d_unique_egdes;
};

template<typename T, typename R = T>
class CubRunLengthEncode : public CubWrapper {
public:
    explicit CubRunLengthEncode(const T* d_in, size_t size,
                                T* d_unique_out, R* d_counts_out) noexcept;
    ~CubRunLengthEncode() noexcept;
    int run() noexcept;
private:
    const T* _d_in;
    T*       _d_unique_out;
    R*       _d_counts_out;
    R*       _d_num_runs_out  { nullptr };
};

template<typename T>
class PartitionFlagged : public CubWrapper {
public:
    explicit PartitionFlagged(const T* d_in, const bool* d_flags,
                              size_t num_items, T* d_out) noexcept;
    ~PartitionFlagged() noexcept;
    int run() noexcept;
private:
    const T*    _d_in;
    const bool* _d_flags;
    T*          _d_out;
    T*          _d_num_selected_out { nullptr };
};

template<typename T>
class CubExclusiveSum : public CubWrapper {
public:
    explicit CubExclusiveSum(T* d_in_out, size_t size)             noexcept;
    explicit CubExclusiveSum(const T* d_in, size_t size, T* d_out) noexcept;
    void run() noexcept;
private:
    const T* _d_in;
    T*       _d_out;
};
/*
template<typename T>
class CubInclusiveSum : public CubWrapper {
public:
    CubInclusiveSum(T* d_in, size_t size);
    CubInclusiveSum(const T* d_in, size_t size, T*& d_out);
    ~CubInclusiveSum() noexcept;
    void run() noexcept;
private:
    static T* null_ptr_ref;
    const T*  _d_in;
    T*&       _d_out;
    T*        _d_in_out;
};
template<typename T>
T* CubInclusiveSum<T>::null_ptr_ref = nullptr;*/

template<typename T>
class CubPartitionFlagged : public CubWrapper {
public:
    CubPartitionFlagged(const T* d_in, const bool* d_flag, size_t size,
                        T*& d_out);
    ~CubPartitionFlagged() noexcept;
    int run() noexcept;
    void run_no_copy() noexcept;
private:
    const T*    _d_in;
    T*&         _d_out;
    const bool* _d_flag;
    int*        _d_num_selected_out;
};

template<typename T>
class CubSegmentedReduce : public CubWrapper {
public:
    CubSegmentedReduce(int* _d_offsets, const T* d_in, int _num_segments,
                       T*& d_out);
    ~CubSegmentedReduce() noexcept;
    void run() noexcept;
private:
    int*  _d_offsets;
    const T*    _d_in;
    T*&         _d_out;
};

template<typename T>
class CubSpMV : public CubWrapper {
public:
    CubSpMV(T* d_value, int* d_row_offsets, int* d_column_indices,
            T* d_vector_x, T* d_vector_y,
            int num_rows, int num_cols, int num_nonzeros);
    //~CubSpMV() noexcept;
    void run() noexcept;
private:
    int*  _d_row_offsets;
    int*  _d_column_indices;
    T*    _d_vector_x;
    T*    _d_vector_y;
    T*    _d_values;
    int   _num_rows, _num_cols, _num_nonzeros;
};

/*
template<typename T>
class CubArgMax : public CubWrapper {
public:
    explicit CubArgMax(const T* d_in, size_t size) noexcept;
    typename std::pair<int, T> run() noexcept;
private:
    const T*                   _d_in;
    cub::KeyValuePair<int, T>* _d_out;
};*/

} // namespace xlib

//#include "impl/CubWrapper.i.cuh"
