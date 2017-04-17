/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
