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
/**
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
#include "Support/Basic.hpp"

#include <csignal>          //std::signal
#include <cstdlib>          //std::exit
#include <iostream>         //std::cout
#if defined(__linux__)
    #include <unistd.h>     //::sysconf
#endif

namespace xlib {

#if defined(__linux__)

void memInfoHost(size_t request) noexcept {
    auto     pages = static_cast<size_t>(::sysconf(_SC_PHYS_PAGES));
    auto page_size = static_cast<size_t>(::sysconf(_SC_PAGE_SIZE));
    detail::memInfoPrint(pages * page_size, pages * page_size - 100u * (1u << 20u),
                 request);
}
#endif

namespace detail {

void memInfoPrint(size_t total, size_t free, size_t request) noexcept {
    ThousandSep sep;
    std::cout << "  Total Memory:\t" << (total >> 20) << " MB\n"
              << "   Free Memory:\t" << (free >> 20)  << " MB\n"
              << "Request memory:\t" << (request >> 20) << " MB\n"
              << "   Request (%):\t" << (request * 100) / total
              << " %\n" << std::endl;
    if (request > free)
        ERROR(" ! Memory too low");
}

} // namespace detail

#if defined(__linux__)

namespace {

[[ noreturn ]] void ctrlC_HandleFun(int) {
    #if defined(__NVCC__)
        cudaDeviceReset();
    #endif
    std::exit(EXIT_FAILURE);
}

} // namespace

void ctrlC_Handle() {
    std::signal(SIGINT, ctrlC_HandleFun);
}

#else

void ctrlC_Handle() {}

#endif

} // namespace xlib
