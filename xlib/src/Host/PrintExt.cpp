/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
#include "Host/PrintExt.hpp"
#include "Host/Basic.hpp"       //xlib::KB
#include "Host/Numeric.hpp"     //xlib::round_div

namespace xlib {

#if defined(__linux__)

std::ostream& operator<<(std::ostream& os, Color mod) {
    return os << "\033[" << static_cast<int>(mod) << "m";
}

std::ostream& operator<<(std::ostream& os, Emph mod) {
    return os << "\033[" << static_cast<int>(mod) << "m";
}

#else

std::ostream& operator<<(std::ostream& os, Color mod) { return os; }
std::ostream& operator<<(std::ostream& os, Emph mod)  { return os; };

#endif

//------------------------------------------------------------------------------

// use space as separator
char myseps::do_thousands_sep() const noexcept { return ','; }
// digits are grouped by 3 digits each
std::string myseps::do_grouping() const noexcept { return "\3"; }

ThousandSep::ThousandSep() : sep(nullptr) {
    sep = new myseps;
    std::cout.imbue(std::locale(std::locale(), sep));
}

ThousandSep::~ThousandSep() {
    std::cout.imbue(std::locale());
}
//------------------------------------------------------------------------------

IosFlagSaver::IosFlagSaver() noexcept : _flags(std::cout.flags()),
                                        _precision(std::cout.precision()) {}

IosFlagSaver::~IosFlagSaver() noexcept {
    std::cout.flags(_flags);
    std::cout.precision(_precision);
}
//------------------------------------------------------------------------------

void fixed_float() noexcept {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
}

void scientific_float() noexcept {
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
}

std::string human_readable(size_t size) noexcept {
    if (size >= xlib::GB)
        return std::to_string(xlib::round_div<xlib::GB>(size)) + " GB";
    if (size >= xlib::MB)
        return std::to_string(xlib::round_div<xlib::MB>(size)) + " MB";
    if (size >= xlib::KB)
        return std::to_string(xlib::round_div<xlib::KB>(size)) + " KB";
    return std::to_string(size) + " B";
}
//------------------------------------------------------------------------------

void char_sequence(char c, int sequence_length) noexcept {
    for (int i = 0; i < sequence_length; i++)
        std::cout << c;
    std::cout << "\n";
}

void printTitle(const std::string& str, char c, int sequence_length) noexcept {
    int dash_size = (sequence_length - static_cast<int>(str.size()) - 1) / 2;
    for (int i = 0; i < dash_size; i++)
        std::cout << c;
    std::cout << " " << str << " ";
    for (int i = 0; i < dash_size; i++)
        std::cout << c;
    if (str.size() % 2 == 1)
        std::cout << c;
    std::cout << "\n";
}
//------------------------------------------------------------------------------

template<>
void printArray<char>(const char* array, size_t size, const std::string& str,
                      const std::string& sep) noexcept {
    std::cout << str;
    if (size == 0)
        std::cout << "empty";
    for (size_t i = 0; i < size; i++)
        std::cout << static_cast<int>(array[i]) << sep;
    std::cout << "\n" << std::endl;
}

template<>
void printArray<unsigned char>(const unsigned char* array, size_t size,
                               const std::string& str,
                               const std::string& sep) noexcept {
    std::cout << str;
    if (size == 0)
        std::cout << "empty";
    for (size_t i = 0; i < size; i++)
        std::cout << static_cast<unsigned>(array[i]) << sep;
    std::cout << "\n" << std::endl;
}

} // namespace xlib
