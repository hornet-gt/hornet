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
/**
 * @version 1.3
 */
namespace xlib {

inline Bitmask::Bitmask(size_t size) noexcept :
                                _size(size),
                                _num_word(xlib::ceil_div<32>(size)) {
    try {
        _array = new unsigned[_num_word]();
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

inline Bitmask::~Bitmask() noexcept {
    delete[] _array;
}

inline void Bitmask::init(size_t size) noexcept {
    _size = size;
    _num_word = xlib::ceil_div<32>(size);
    try {
        _array = new unsigned[_num_word]();
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

inline void Bitmask::free() noexcept {
    delete[] _array;
    _array = nullptr;
}

inline BitRef Bitmask::operator[](size_t index) noexcept {
    assert(index < _size);
    return BitRef(_array[index >> 5u], 1u << (index % 32u));
}

inline bool Bitmask::operator[](size_t index) const noexcept {
    assert(index < _size);
    return static_cast<bool>(_array[index >> 5u] & (1u << (index % 32u)));
}

inline void Bitmask::clear() noexcept {
    std::fill(_array, _array + _num_word, 0);
}

inline size_t Bitmask::get_count() const noexcept {
    size_t count = 0;
    for (size_t i = 0; i < _num_word; i++)
        count += static_cast<size_t>(__builtin_popcount(_array[i]));
    return count;
}

//==============================================================================

template<unsigned SIZE>
inline BitRef BitmaskStack<SIZE>::operator[](unsigned index) noexcept {
    assert(index < SIZE);
    return BitRef(_array[index >> 5u], 1u << (index % 32u));
}

template<unsigned SIZE>
inline bool BitmaskStack<SIZE>::operator[](unsigned index) const noexcept {
    assert(index < SIZE);
    return static_cast<bool>(_array[index >> 5u] & (1u << (index % 32u)));
}

template<unsigned SIZE>
inline void BitmaskStack<SIZE>::clear() noexcept {
    std::fill(_array, _array + xlib::CeilDiv<SIZE, 32>::value, 0);
}

template<unsigned SIZE>
inline unsigned BitmaskStack<SIZE>::get_count() const noexcept {
    unsigned count = 0;
    for (unsigned i = 0; i < xlib::CeilDiv<SIZE, 32>::value; i++)
        count += static_cast<unsigned>(__builtin_popcount(_array[i]));
    return count;
}

} // namespace xlib
