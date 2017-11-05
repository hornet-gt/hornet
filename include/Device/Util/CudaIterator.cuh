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

/**
 * offset iterator
 */
template<typename T, unsigned BlockSize>
struct cuda_iterator {
protected:
    cusize_t offset;
    const cusize_t stride = BlockSize * gridDim.x;
public:
    __device__ __forceinline__
    cuda_iterator(cusize_t _offset) : offset(_offset) {}

    /*__device__ __forceinline__
    cuda_iterator(const cuda_iterator<T, BlockSize>& obj) = delete;*/

    __device__ __forceinline__
    cusize_t operator*() const {
        return offset;
    }

    __device__ __forceinline__
    cuda_iterator& operator++() {
        offset += stride;
        return *this;
    }

    __device__ __forceinline__
    bool operator!=(const cuda_iterator& obj) const {
        return offset < obj.offset;
    }
};


/**
 * pointer iterator, safe for warp/block (all threads enter in the main loop)
 */
template<unsigned BlockSize, typename iterator_node_t>
struct cuda_safe_iterator : cuda_iterator<cusize_t, BlockSize> {
protected:
    cusize_t max_size;
public:
    iterator_node_t& node;
    using cuda_iterator<cusize_t, BlockSize>::offset;

    __device__ __forceinline__
    cuda_safe_iterator(cusize_t _offset,
                       cusize_t _max_size,
                       iterator_node_t& _node) :
        cuda_iterator<cusize_t, BlockSize>(_offset),
        max_size(_max_size),
        node(_node) {}

    /*__device__ __forceinline__
    cuda_safe_iterator(const cuda_safe_iterator
                            <BlockSize, iterator_node_t>& obj) = delete;*/

    __device__ __forceinline__
    iterator_node_t& operator*() {
        node.eval(offset, max_size);
        return node;
    }
};

//------------------------------------------------------------------------------

/**
 * pointer + offset iterator
 */
template<typename T, unsigned BlockSize>
struct cuda_forward_iterator {
protected:
    T* const ptr = nullptr;
public:
    cusize_t offset;
    const cusize_t stride = BlockSize * gridDim.x;

    __device__ __forceinline__
    cuda_forward_iterator() : offset(0) {}

    __device__ __forceinline__
    cuda_forward_iterator(T* const _ptr, cusize_t _offset) :
        ptr(_ptr),
        offset(_offset) {}

    /*__device__ __forceinline__
    cuda_forward_iterator(const cuda_forward_iterator<T, BlockSize>& obj)
        = delete;*/

    __device__ __forceinline__
    cuda_forward_iterator& operator++() {
        offset += stride;
        return *this;
    }

    __device__ __forceinline__
    T& operator*() const {
        return *(ptr + offset);
    }

    __device__ __forceinline__
    bool operator!=(const cuda_forward_iterator& obj) const {
        return obj.offset > offset;
    }
};

//==============================================================================

/**
 * pointer + offset data structure
 */
template<typename T, unsigned BlockSize, unsigned VW_SIZE = 1>
struct cu_array {
private:
    T* const array;
    const cusize_t size;
    static const unsigned STRIDE = BlockSize / VW_SIZE;
public:
    __device__ __forceinline__
    cu_array(T* _array, T _size) : array(_array), size(_size) {}

    __device__ __forceinline__
    cuda_forward_iterator<T, STRIDE> begin() const {
        const unsigned global_id = (blockIdx.x * BlockSize + threadIdx.x)
                                   / VW_SIZE;
        return cuda_forward_iterator<T, STRIDE>(array, global_id);
    }

    __device__ __forceinline__
    cuda_forward_iterator<T, STRIDE> end() const {
        return cuda_forward_iterator<T, STRIDE>(array, size);
    }
};

template<typename T, unsigned BlockSize, unsigned VW_SIZE = 1>
struct cuda_range_loop {
private:
    const cusize_t size;
    static const unsigned STRIDE = BlockSize / VW_SIZE;
public:
    __device__ __forceinline__
    cuda_range_loop(T _size) : size(_size) {}

    __device__ __forceinline__
    cuda_iterator<T, STRIDE> begin() const {
        const unsigned global_id = (blockIdx.x * BlockSize + threadIdx.x)
                                   / VW_SIZE;
        return cuda_iterator<T, STRIDE>(global_id);
    }

    __device__ __forceinline__
    cuda_iterator<T, STRIDE> end() const {
        return cuda_iterator<T, STRIDE>(size);
    }
};

} // namespace xlib
