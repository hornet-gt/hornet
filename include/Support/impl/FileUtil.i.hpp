/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

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
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
#include "Support/Basic.hpp"
#include "Support/Numeric.hpp" //xlib::per_cent
#include <cassert>               //assert
#include <cmath>                 //std::round
#include <iomanip>               //std::setw

#if defined(__linux__)
    #include <fcntl.h>          //::open
    #include <sys/mman.h>       //::mmap
    #include <sys/stat.h>       //::open
    #include <sys/types.h>      //::open
    #include <unistd.h>         //::lseek
#endif

namespace xlib {

inline Progress::Progress(size_t total) noexcept :
                     _float_chunk(static_cast<double>(total - 1) / 100.0),
                     _total(total),
                     _next_chunk(static_cast<size_t>(_float_chunk)),
                     _to_print(1) {}

inline void Progress::next(size_t progress) noexcept {
    if (progress == 0) {
        std::cout << ((_next_chunk == 0) ? "   100%\n" : "     0%")
                  << std::flush;
    }
    else if (progress == _next_chunk) {
        std::cout << "\b\b\b\b\b\b\b" << std::setw(6) << _to_print++
                  << "%" << std::flush;
        _next_chunk = static_cast<size_t>(
                                 static_cast<double>(_to_print) * _float_chunk);
        if (_to_print == 101)
            std::cout << std::endl;
    }
}

inline void Progress::per_cent(size_t progress) const noexcept {
    if (progress == 0) {
        std::cout << "     0%" << std::flush;
        return;
    }
    std::cout << "\b\b\b\b\b\b\b" << std::setw(6)
           << std::round(xlib::per_cent(progress, _total)) << "%" << std::flush;
    if (progress == _total)
        std::cout << std::endl;
}

//------------------------------------------------------------------------------

#if defined(__linux__)

inline MemoryMapped::MemoryMapped(const char* filename, size_t file_size,
                                  Enum mode, bool print) noexcept :
                                    _progress(file_size),
                                    _file_size(file_size),
                                    _print(print) {

    _fd = mode == READ ? ::open(filename, O_RDONLY, S_IRUSR) :
                ::open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (_fd == -1) ERROR("::open")

    if (::lseek(_fd, static_cast<off_t>(file_size - 1), SEEK_SET) == -1)
        ERROR("::lseek")
    if (mode == WRITE && ::write(_fd, "", 1) != 1)
        ERROR("::write")

    _mmap_ptr = static_cast<char*>(::mmap(nullptr, file_size,
                                          mode == READ ? PROT_READ : PROT_WRITE,
                                          MAP_SHARED, _fd, 0));
    if (_mmap_ptr == MAP_FAILED) ERROR("::mmap");
    if (::madvise(_mmap_ptr, file_size, MADV_SEQUENTIAL) == -1)
        ERROR("::madvise");
}

inline MemoryMapped::~MemoryMapped() noexcept {
    if (::munmap(_mmap_ptr, _file_size) == -1) ERROR("::munmap");
    if (::close(_fd) == -1) ERROR("::close");
}

template<typename, typename... Ts>
void MemoryMapped::write() const noexcept {
    if (_print)
        _progress.per_cent(_partial);
}

template<typename, typename... Ts>
void MemoryMapped::read() const noexcept {
    if (_print)
        _progress.per_cent(_partial);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

template<typename T, typename... Ts>
void MemoryMapped::write(const T* data, size_t size, Ts... args) {
    if (_print)
        _progress.per_cent(_partial);
    std::copy(data, data + size,                                        //NOLINT
              reinterpret_cast<T*>(_mmap_ptr + _partial));              //NOLINT
    _partial += size * sizeof(T);
    assert(_partial <= _file_size);
    write(args...);
}

template<typename T, typename... Ts>
void MemoryMapped::read(T* data, size_t size, Ts... args) {
    if (_print)
        _progress.per_cent(_partial);
    std::copy(reinterpret_cast<T*>(_mmap_ptr + _partial),               //NOLINT
              reinterpret_cast<T*>(_mmap_ptr + _partial) + size, data); //NOLINT
    _partial += size * sizeof(T);

    read(args...);
}

#pragma clang diagnostic pop
#endif

} // namespace xlib
