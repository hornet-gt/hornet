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

#include <cstddef>  //size_t
#include <fstream>  //std::ifstream
#include <string>   //std::string

namespace xlib {

class Progress {
public:
    explicit Progress(size_t total) noexcept;
    void     next    (size_t progress) noexcept;
    void     per_cent(size_t progress) const noexcept;
private:
    const double _float_chunk;
    const size_t _total;
    size_t       _next_chunk;
    int          _to_print;
};

#if defined(__linux__)

class MemoryMapped {
public:
    enum Enum { READ, WRITE };
    explicit MemoryMapped(const char* filename, size_t file_size, Enum mode,
                          bool print = false) noexcept;
    ~MemoryMapped() noexcept;

    template<typename T, typename... Ts>
    void read(T* data, size_t size, Ts... args);

    template<typename T, typename... Ts>
    void write(const T* data, size_t size, Ts... args);

private:
    template<typename = void, typename... Ts>
    void read() const noexcept;

    template<typename = void, typename... Ts>
    void write() const noexcept;

    Progress _progress;
    char*    _mmap_ptr   { nullptr };
    size_t   _partial    {};
    size_t   _file_size;
    int      _fd         {};
    bool     _print;
};

#endif

void        check_regular_file(const char* filename);
void        check_regular_file(std::ifstream& fin, const char* filename = "");
size_t      file_size(const char* filename);
size_t      file_size(std::ifstream& fin);

std::string extract_filename            (const std::string& str) noexcept;
std::string extract_file_extension      (const std::string& str) noexcept;
std::string extract_filepath            (const std::string& str) noexcept;
std::string extract_filepath_noextension(const std::string& str) noexcept;
void        skip_lines(std::istream& fin, int num_lines = 1);
void        skip_words(std::istream& fin, int num_words = 1);

} // namespace xlib

#include "impl/FileUtil.i.hpp"
