/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
#ifndef BITTREE_I_CUH
#define BITTREE_I_CUH

#include <Host/Numeric.hpp>             //xlib::ceil_log
#include <iterator>                     //std::distance

namespace hornet {

#define BITREE BitTree<degree_t>

template <typename degree_t>
BITREE::
BitTree(degree_t block_items, degree_t blockarray_items) noexcept :
        _block_items(block_items),
        _log_block_items(xlib::log2(block_items)),
        _blockarray_items(blockarray_items),
        _num_blocks(_blockarray_items / _block_items),
        _num_levels(xlib::max(xlib::ceil_log<WORD_SIZE>(_num_blocks), 1)),
        _internal_bits(xlib::geometric_serie<WORD_SIZE>(_num_levels - 1) - 1),
        _internal_words(xlib::ceil_div<WORD_SIZE>(_internal_bits)),
        _external_words(xlib::ceil_div<WORD_SIZE>(_num_blocks)),
        _num_words(_internal_words + _external_words),
        _total_bits(_num_words * WORD_SIZE),
        _array(new word_t[_num_words]),
        _last_level(_array + _internal_words),
        _size(0) {

    assert(xlib::is_power2(block_items));
    assert(xlib::is_power2(blockarray_items));
    assert(block_items <= blockarray_items);

    const word_t EMPTY = static_cast<word_t>(-1);
    std::fill(_array, _array + _num_words, EMPTY);
}

template <typename degree_t>
BITREE::
~BitTree(void) {
    delete[] _array;
}

template <typename degree_t>
BITREE::
BitTree(BITREE&& obj) noexcept :
                    _block_items(obj._block_items),
                    _log_block_items(obj._log_block_items),
                    _blockarray_items(obj._blockarray_items),
                    _num_blocks(obj._num_blocks),
                    _num_levels(obj._num_levels),
                    _internal_bits(obj._internal_bits),
                    _internal_words(obj._internal_words),
                    _external_words(obj._external_words),
                    _num_words(obj._num_words),
                    _total_bits(obj._total_bits),
                    _array(obj._array),
                    _last_level(_array + _internal_words),
                    _size(obj._size) {

    obj._array      = nullptr;
    obj._last_level = nullptr;
    obj._size       = 0;
}

template <typename degree_t>
BITREE::
BitTree(const BITREE& obj) noexcept :
                    _block_items(obj._block_items),
                    _log_block_items(obj._log_block_items),
                    _blockarray_items(obj._blockarray_items),
                    _num_blocks(obj._num_blocks),
                    _num_levels(obj._num_levels),
                    _internal_bits(obj._internal_bits),
                    _internal_words(obj._internal_words),
                    _external_words(obj._external_words),
                    _num_words(obj._num_words),
                    _total_bits(obj._total_bits),
                    _array(new word_t[_num_words]),
                    _last_level(_array + _internal_words),
                    _size(obj._size) {
    std::copy(obj._array, obj._array + _num_words, _array);
}

template <typename degree_t>
BITREE&
BITREE::
operator=(BitTree&& obj) noexcept {
    assert( (_block_items == obj._block_items) &&
            (_blockarray_items == obj._blockarray_items) );
    delete[] _array;
    _array          = obj._array;
    _last_level     = obj._last_level;
    _size           = obj._size;

    obj._array      = nullptr;
    obj._last_level = nullptr;
    obj._size       = 0;
    return *this;
}

template <typename degree_t>
BITREE&
BITREE::
operator=(const BitTree& obj) noexcept {
    assert( (_block_items == obj._block_items) &&
            (_blockarray_items == obj._blockarray_items) );
    std::copy(obj._array, obj._array + _num_words, _array);
    _size           = obj._size;
    return *this;
}

//------------------------------------------------------------------------------

template <typename degree_t>
degree_t
BITREE::
insert() noexcept {
    assert(_size < _num_blocks && "tree is full");
    _size++;
    //find the first empty location
    degree_t index = 0;
    for (degree_t i = 0; i < _num_levels - 1; i++) {
        assert(index < _total_bits && _array[index / WORD_SIZE] != 0);
        degree_t pos = __builtin_ctz(_array[index / WORD_SIZE]);
        index   = (index + pos + 1) * WORD_SIZE;
    }
    assert(index < _total_bits && _array[index / WORD_SIZE] != 0);
    index += __builtin_ctz(_array[index / WORD_SIZE]);
    assert(index < _total_bits);

    xlib::delete_bit(_array, index);
    if (_array[index / WORD_SIZE] == 0) {
        const auto& lambda = [&](degree_t index) {
                                          xlib::delete_bit(_array, index);
                                          return _array[index / WORD_SIZE] != 0;
                                        };
        parent_traverse(index, lambda);
    }
    degree_t block_index = index - _internal_bits;
    assert(block_index >= 0 && block_index < _blockarray_items);
    return block_index;
}

//------------------------------------------------------------------------------
template <typename degree_t>
void
BITREE::
remove(degree_t diff) noexcept {
    unsigned _diff = diff;
    assert(_size != 0 && "tree is empty");
    _size--;
    degree_t p_index = _diff >> _log_block_items;   // diff / block_items
    assert(p_index < _external_words * sizeof(word_t) * 8u);
    assert(xlib::read_bit(_last_level, p_index) == 0 && "not found");
    xlib::write_bit(_last_level, p_index);
    p_index += _internal_bits;

    parent_traverse(p_index, [&](degree_t index) {
                                bool ret = _array[index / WORD_SIZE] != 0;
                                xlib::write_bit(_array, index);
                                return ret;
                            });
}

template <typename degree_t>
template<typename Lambda>
void BITREE::
parent_traverse(degree_t index, const Lambda& lambda) noexcept {
    index /= WORD_SIZE;
    while (index != 0) {
        index--;
        if (lambda(index))
            return;
        index /= WORD_SIZE;
    }
}

template <typename degree_t>
degree_t
BITREE::
size() const noexcept {
    return _size;
}

template <typename degree_t>
bool
BITREE::
full() const noexcept {
    return _size == _num_blocks;
}

template <typename degree_t>
void
BITREE::
print() const noexcept {
    const degree_t ROW_SIZE = 64;
    degree_t          count = WORD_SIZE;
    auto       tmp_ptr = _array;
    std::cout << "BitTree:\n";

    for (degree_t i = 0; i < _num_levels - 1; i++) {
        std::cout << "\nlevel " << i << " :\n";
        assert(count < ROW_SIZE || count % ROW_SIZE == 0);

        degree_t size = std::min(count, ROW_SIZE);
        for (degree_t j = 0; j < count; j += ROW_SIZE) {
            xlib::printBits(tmp_ptr, size);
            tmp_ptr += size / WORD_SIZE;
            if (tmp_ptr >= _array + _num_words)
                break;
        }
        count *= WORD_SIZE;
    }
    std::cout << "\nlevel " << _num_levels - 1 << " :\n";
    xlib::printBits(tmp_ptr, _external_words * WORD_SIZE);
    std::cout << std::endl;
}

template <typename degree_t>
void
BITREE::
statistics() const noexcept {
    std::cout << "\nBitTree Statistics:\n"
              << "\n     BLOCK_ITEMS: " << _block_items
              << "\nBLOCKARRAY_ITEMS: " << _blockarray_items
              << "\n      NUM_BLOCKS: " << _num_blocks
              //<< "\n       sizeof(T): " << sizeof(block_t)
              //<< "\n   BLOCK_SIZE(b): " << _block_items * sizeof(block_t)
              << "\n"
              << "\n      NUM_LEVELS: " << _num_levels
              << "\n       WORD_SIZE: " << WORD_SIZE
              << "\n   INTERNAL_BITS: " << _internal_bits
              << "\n   EXTERNAL_BITS: " << _num_blocks
              << "\n      TOTAL_BITS: " << _total_bits
              << "\n  INTERNAL_WORDS: " << _internal_words
              << "\n EXTERNAL_WORLDS: " << _external_words
              << "\n       NUM_WORDS: " << _num_words << "\n\n";
}

template <typename degree_t>
degree_t
BITREE::
get_log_block_items() const noexcept {
    return _log_block_items;
}

//==============================================================================

}
#endif
