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
#include "Support/PrintExt.hpp"

namespace xlib {

#if defined(__linux__)

std::ostream& operator<<(std::ostream& os, const Color& mod) {
    return os << "\033[" << static_cast<int>(mod) << "m";
}

std::ostream& operator<<(std::ostream& os, const Emph& mod) {
    return os << "\033[" << static_cast<int>(mod) << "m";
}

#else

std::ostream& operator<<(std::ostream& os, const Color& mod) { return os; }
std::ostream& operator<<(std::ostream& os, const Emph& mod)  { return os; };

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

IosFlagSaver::IosFlagSaver() noexcept : _flags(std::cout.flags()),
                                        _precision(std::cout.precision()) {}

IosFlagSaver::~IosFlagSaver() noexcept {
    std::cout.flags(_flags);
    std::cout.precision(_precision);
}

void fixedFloat() {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
}

void scientificFloat() {
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
}

//------------------------------------------------------------------------------

void charSequence(char c, int sequence_length) noexcept {
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
void printArray<char>(char* array, size_t size, const std::string& str,
                      char sep) noexcept {
    std::cout << str;
    for (size_t i = 0; i < size; i++)
        std::cout << static_cast<int>(array[i]) << sep;
    std::cout << "\n" << std::endl;
}

template<>
void printArray<unsigned char>(unsigned char* array, size_t size,
                               const std::string& str, char sep) noexcept {
    std::cout << str;
    for (size_t i = 0; i < size; i++)
        std::cout << static_cast<unsigned>(array[i]) << sep;
    std::cout << "\n" << std::endl;
}

} // namespace xlib
