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

#include "Base/Device/Util/CacheModifier.cuh"

namespace xlib {

enum class cuQUEUE_MODE { SIMPLE, UNROLL, Min, SHAREDMEM,
                          SHAREDMEM_UNROLL, BALLOT};

template<cuQUEUE_MODE mode = cuQUEUE_MODE::SIMPLE,
         CacheModifier CM = DF, int Items_per_warp = 0>
struct warp_dyn {
    template<typename T, int SIZE>
     __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            int& size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            const int total,          //optional
                            T* __restrict__ SMem);    //optional
};

template<cuQUEUE_MODE mode = cuQUEUE_MODE::SIMPLE,
         CacheModifier CM = DF, int Items_per_warp = 0>
struct QueueWarp {
     template<typename T, typename R, int SIZE>
     __device__ __forceinline__
    static void store(T (&Queue)[SIZE],
                      int size,
                      T* __restrict__ queue_ptr,
                      R* __restrict__ queue_size_ptr,
                      T* __restrict__ SMem);

    template<typename T, typename R, int SIZE>
    __device__ __forceinline__
    static void store(T (&Queue)[SIZE],
                      int size,
                      T* __restrict__ queue_ptr,
                      R* __restrict__ queue_size_ptr);

    /*template<typename T, typename R, int SIZE>
    __device__ __forceinline__
    static void store2(T (&Queue)[SIZE],
                       const int size,
                       T* __restrict__ queue_ptr,
                       R* __restrict__ queue_size_ptr,
                       T* __restrict__ SMem,
                       int& warp_offset,
                       int& total);*/
};

//==============================================================================
namespace detail {

template<typename T, int SIZE, CacheModifier CM = DF>
class WarpQueueBase {
public:
    __device__ __forceinline__
    WarpQueueBase(T (&queue)[SIZE],
                  T*   __restrict__ queue_ptr,
                  int* __restrict__ size_ptr);
protected:
    T (&_queue)[SIZE];
    T*   _queue_ptr;
    int* _size_ptr;
    int  _size;
};

} // namespace detail

template<typename T, int SIZE, CacheModifier CM = DF>
class WarpQueueSimple : public detail::WarpQueueBase<T, SIZE, CM> {
public:
    __device__ __forceinline__
    WarpQueueSimple(T (&queue)[SIZE],
                   T*   __restrict__ queue_ptr,
                   int* __restrict__ size_ptr);

    __device__ __forceinline__
    ~WarpQueueSimple();

    __device__ __forceinline__
    void insert(T item);

    __device__ __forceinline__
    void store();
private:
    using detail::WarpQueueBase<T, SIZE, CM>::_queue;
    using detail::WarpQueueBase<T, SIZE, CM>::_queue_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size;
    __device__ __forceinline__
    void _store();
};

template<typename T, int SIZE, CacheModifier CM = DF>
class WarpQueueUnroll : public detail::WarpQueueBase<T, SIZE, CM> {
public:
    __device__ __forceinline__
    WarpQueueUnroll(T (&queue)[SIZE],
                    T*   __restrict__ queue_ptr,
                    int* __restrict__ size_ptr);

    __device__ __forceinline__
    ~WarpQueueUnroll();

    __device__ __forceinline__
    void insert(T item);

    __device__ __forceinline__
    void store();
private:
    using detail::WarpQueueBase<T, SIZE, CM>::_queue;
    using detail::WarpQueueBase<T, SIZE, CM>::_queue_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size;
    __device__ __forceinline__
    void _store();
};


template<typename T, CacheModifier CM = DF>
class WarpQueueBallot {
public:
    __device__ __forceinline__
    WarpQueueBallot(T*   __restrict__ queue_ptr,
                    int* __restrict__ size_ptr);

    __device__ __forceinline__
    void store(T item, int predicate);
private:
    T*   _queue_ptr;
    int* _size_ptr;
};

template<typename T, int SIZE, unsigned ITEMS_PER_WARP, CacheModifier CM = DF>
class WarpQueueSharedMem : public detail::WarpQueueBase<T, SIZE, CM> {
public:
    __device__ __forceinline__
    WarpQueueSharedMem(T (&queue)[SIZE],
                       T*   __restrict__ queue_ptr,
                       int* __restrict__ size_ptr,
                       T* shared_mem);

    __device__ __forceinline__
    ~WarpQueueSharedMem();

    __device__ __forceinline__
    void insert(T item);

    __device__ __forceinline__
    void store();
private:
    using detail::WarpQueueBase<T, SIZE, CM>::_queue;
    using detail::WarpQueueBase<T, SIZE, CM>::_queue_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size;
    T* _shared_mem;
    T* _lane_shared_mem;
};

} // namespace xlib

#include "impl/WarpDynamic.i.cuh"
