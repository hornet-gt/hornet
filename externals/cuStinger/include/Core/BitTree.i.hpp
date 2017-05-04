/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
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
#pragma once

#include "Support/Host/Numeric.hpp"
#include "Support/Device/SafeCudaAPI.cuh"

namespace custinger {

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::BitTree() noexcept :
                                              _last_level(nullptr),
                                              _h_ptr(nullptr),
                                              _d_ptr(nullptr),
                                              _size(0) {
    _h_ptr = new T[BLOCKARRAY_ITEMS];
    cuMalloc(_d_ptr, BLOCKARRAY_ITEMS);
    const word_t EMPTY = static_cast<word_t>(-1);
    std::fill(_array, _array + NUM_WORDS, EMPTY);
    _last_level = _array + INTERNAL_WORDS;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>
::BitTree(BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>&& obj) noexcept :
                                _last_level(_array + INTERNAL_WORDS),
                                _h_ptr(obj._h_ptr),
                                _d_ptr(obj._d_ptr),
                                _size(obj._size) {

    std::copy(obj._array, obj._array + NUM_WORDS, _array);
    obj._last_level = nullptr;
    obj._h_ptr      = nullptr;
    obj._d_ptr      = nullptr;
    obj._size       = 0;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>&
BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>
::operator=(BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>&& obj) noexcept {
    std::copy(obj._array, obj._array + NUM_WORDS, _array);
    _last_level = _array + INTERNAL_WORDS;
    _d_ptr   = obj._d_ptr;
    _h_ptr   = obj._h_ptr;
    _size    = obj._size;

    obj._last_level = nullptr;
    obj._h_ptr      = nullptr;
    obj._d_ptr      = nullptr;
    obj._size       = 0;
    return *this;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::~BitTree() noexcept {
    cuFree(_d_ptr);
    delete[] _h_ptr;
    _d_ptr = nullptr;

}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
void BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::free_host_ptr() noexcept {
    delete[] _h_ptr;
    _h_ptr = nullptr;
}

//------------------------------------------------------------------------------

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
inline std::pair<T*, T*>
BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::insert() noexcept {
    assert(_size < NUM_BLOCKS && "tree is full");
    _size++;
    //find the first empty location
    int index = 0;
    for (int i = 0; i < NUM_LEVELS - 1; i++) {
        assert(index < TOTAL_BITS && _array[index / WORD_SIZE] != 0);
        int pos = __builtin_ctz(_array[index / WORD_SIZE]);
        index   = (index + pos + 1) * WORD_SIZE;
    }
    assert(index < TOTAL_BITS && _array[index / WORD_SIZE] != 0);
    index += __builtin_ctz(_array[index / WORD_SIZE]);
    assert(index < TOTAL_BITS);

    xlib::delete_bit(_array, index);
    if (_array[index / WORD_SIZE] == 0) {
        auto lambda = [&](int index) { xlib::delete_bit(_array, index);
                                       return _array[index / WORD_SIZE] != 0;
                                     };
        parent_traverse(index, lambda);
    }
    int block_index = index - INTERNAL_BITS;
    return std::pair<T*, T*>(_h_ptr + block_index * BLOCK_ITEMS,
                             _d_ptr + block_index * BLOCK_ITEMS);
}

//------------------------------------------------------------------------------

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
inline void BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>
::remove(T* to_delete) noexcept {
    assert(_size != 0 && "tree is empty");
    _size--;
    int p_index = remove_aux(to_delete) + INTERNAL_BITS;
    parent_traverse(p_index, [&](int index) {
            bool ret = _array[index / WORD_SIZE] != 0;
            xlib::write_bit(_array, index);
            return ret;
        });
}

//------------------------------------------------------------------------------

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
inline int BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>
::remove_aux(T* to_delete) noexcept {
    unsigned diff = std::distance(_d_ptr, to_delete);
    int     index = diff / BLOCK_ITEMS;
    assert(index < NUM_BLOCKS);
    assert(xlib::read_bit(_last_level, index) == 0 && "not found");

    xlib::write_bit(_last_level, index);
    return index;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
template<typename Lambda>
inline void BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>
::parent_traverse(int index, const Lambda& lambda) noexcept {
    index /= WORD_SIZE;
    while (index != 0) {
        index--;
        if (lambda(index))
            return;
        index /= WORD_SIZE;
    }
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
inline int BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::size() const noexcept {
    return _size;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
inline bool BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>
::is_full() const noexcept {
    return _size == NUM_BLOCKS;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
inline std::pair<T*, T*>
BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::base_address() const noexcept {
    return std::pair<T*, T*>(_h_ptr, _d_ptr);
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
inline bool BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>
::belong_to(T* to_check) const noexcept {
    return to_check >= _d_ptr && to_check < _d_ptr + BLOCKARRAY_ITEMS;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
void BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::print() const noexcept {
    const int ROW_SIZE = 64;
    int          count = WORD_SIZE;
    auto       tmp_ptr = _array;
    std::cout << "BitTree:\n";

    for (int i = 0; i < NUM_LEVELS - 1; i++) {
        std::cout << "\nlevel " << i << " :\n";
        assert(count < ROW_SIZE || count % ROW_SIZE == 0);

        int size = std::min(count, ROW_SIZE);
        for (int j = 0; j < count; j += ROW_SIZE) {
            xlib::printBits(tmp_ptr, size);
            tmp_ptr += size / WORD_SIZE;
            if (tmp_ptr >= _array + NUM_WORDS)
                break;
        }
        count *= WORD_SIZE;
    }
    std::cout << "\nlevel " << NUM_LEVELS - 1 << " :\n";
    xlib::printBits(tmp_ptr, EXTERNAL_WORLDS * WORD_SIZE);
    std::cout << std::endl;
}

template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
void BitTree<T, BLOCK_ITEMS, BLOCKARRAY_ITEMS>::statistics() const noexcept {
    std::cout << "\nBitTree Statistics:\n"
              << "\n     BLOCK_ITEMS: " << BLOCK_ITEMS
              << "\nBLOCKARRAY_ITEMS: " << BLOCKARRAY_ITEMS
              << "\n      NUM_BLOCKS: " << NUM_BLOCKS
              << "\n          type T: " << xlib::type_name<T>()
              << "\n       sizeof(T): " << sizeof(T)
              << "\n   BLOCK_SIZE(b): " << BLOCK_ITEMS * sizeof(T) << "\n"
              << "\n      NUM_LEVELS: " << NUM_LEVELS
              << "\n       WORD_SIZE: " << WORD_SIZE
              << "\n   INTERNAL_BITS: " << INTERNAL_BITS
              << "\n   EXTERNAL_BITS: " << NUM_BLOCKS
              << "\n      TOTAL_BITS: " << TOTAL_BITS
              << "\n  INTERNAL_WORDS: " << INTERNAL_WORDS
              << "\n EXTERNAL_WORLDS: " << EXTERNAL_WORLDS
              << "\n       NUM_WORDS: " << NUM_WORDS << "\n\n";
}

} // namespace custinger
