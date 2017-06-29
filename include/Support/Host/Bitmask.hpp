/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Support/Host/BitRef.hpp"
#include "Support/Host/Numeric.hpp"
#include <cstddef>

namespace xlib {

class Bitmask {
public:
    explicit Bitmask()            noexcept = default;
    explicit Bitmask(size_t size) noexcept;
    ~Bitmask() noexcept;

    void init(size_t size) noexcept;
    void free()            noexcept;

    bool   operator[](size_t index) const noexcept;
    BitRef operator[](size_t index) noexcept;

    void   randomize()              noexcept;
    void   randomize(uint64_t seed) noexcept;
    void   clear()                  noexcept;
    size_t size()                   const noexcept;

    Bitmask(const Bitmask&)        = delete;
    bool operator=(const Bitmask&) = delete;
private:
    size_t    _num_word { 0 };
    size_t    _size     { 0 };
    unsigned* _array    { nullptr };
};

template<unsigned SIZE>
class BitmaskStack {
public:
    explicit BitmaskStack() noexcept = default;

    bool   operator[](unsigned index) const noexcept;
    BitRef operator[](unsigned index) noexcept;

    void     clear()     noexcept;
    unsigned get_count() const noexcept;

    BitmaskStack(const BitmaskStack&)   = delete;
    bool operator=(const BitmaskStack&) = delete;
private:
    unsigned _array[xlib::CeilDiv<SIZE, 32>::value] = {};
};

} // namespace xlib

#include "impl/Bitmask.i.hpp"
