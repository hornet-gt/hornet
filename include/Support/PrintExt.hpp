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

#include "Support/HostDevice.hpp"
#include <iostream> //std::numpunct
#include <string>   //std::string

namespace xlib {

/**
 * @brief change the color of the output stream
 */
enum class Color {
                       /** <table border="0"><tr><td><div> Red </div></td><td><div style="background:#FF0000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_RED       = 31, /** <table border="0"><tr><td><div> Green </div></td><td><div style="background:#008000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_GREEN     = 32, /** <table border="0"><tr><td><div> Yellow </div></td><td><div style="background:#FFFF00;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_YELLOW    = 33, /** <table border="0"><tr><td><div> Blue </div></td><td><div style="background:#0000FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_BLUE      = 34, /** <table border="0"><tr><td><div> Magenta </div></td><td><div style="background:#FF00FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_MAGENTA   = 35, /** <table border="0"><tr><td><div> Cyan </div></td><td><div style="background:#00FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_CYAN      = 36, /** <table border="0"><tr><td><div> Light Gray </div></td><td><div style="background:#D3D3D3;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_GRAY    = 37, /** <table border="0"><tr><td><div> Dark Gray </div></td><td><div style="background:#A9A9A9;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_D_GREY    = 90, /** <table border="0"><tr><td><div> Light Red </div></td><td><div style="background:#DC143C;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_RED     = 91, /** <table border="0"><tr><td><div> Light Green </div></td><td><div style="background:#90EE90;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_GREEN   = 92, /** <table border="0"><tr><td><div> Light Yellow </div></td><td><div style="background:#FFFFE0;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_YELLOW  = 93, /** <table border="0"><tr><td><div> Light Blue </div></td><td><div style="background:#ADD8E6;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_BLUE    = 94, /** <table border="0"><tr><td><div> Light Magenta </div></td><td><div style="background:#EE82EE;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_MAGENTA = 95, /** <table border="0"><tr><td><div> Light Cyan </div></td><td><div style="background:#E0FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_CYAN    = 96, /** <table border="0"><tr><td><div> White </div></td><td><div style="background:#FFFFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_WHITE     = 97, /** Default */
    FG_DEFAULT   = 39
};

/**
 * @enum Emph
 */
enum class Emph {
    SET_BOLD      = 1,
    SET_DIM       = 2,
    SET_UNDERLINE = 4,
    SET_RESET     = 0,
};

/// @cond
std::ostream& operator<<(std::ostream& os, const Color& mod);
std::ostream& operator<<(std::ostream& os, const Emph& mod);
/// @endcond
//------------------------------------------------------------------------------

struct myseps : std::numpunct<char> {
private:
    char do_thousands_sep()   const noexcept final;
    std::string do_grouping() const noexcept final;
};

class ThousandSep {
public:
    ThousandSep();
    ~ThousandSep();

    ThousandSep(const ThousandSep&)    = delete;
    void operator=(const ThousandSep&) = delete;
private:
    myseps* sep { nullptr };
};

template<typename T>
inline std::string format(T num, unsigned precision = 1) noexcept;

void fixedFloat();
void scientificFloat();

class IosFlagSaver {
public:
    IosFlagSaver()  noexcept;
    ~IosFlagSaver() noexcept;
    IosFlagSaver(const IosFlagSaver &rhs)             = delete;
    IosFlagSaver& operator= (const IosFlagSaver& rhs) = delete;
private:
    std::ios::fmtflags _flags;
    std::streamsize    _precision;
};
//------------------------------------------------------------------------------

void charSequence(char c, int sequence_length = 80) noexcept;

void printTitle(const std::string& str, char c = '-',
                int sequence_length = 80) noexcept;
//------------------------------------------------------------------------------

template<typename T, int SIZE>
void printArray(T (&array)[SIZE], const std::string& str = "", char sep = ' ')
                noexcept;

template<typename T>
void printArray(T* array, size_t size, const std::string& str = "",
                char sep = ' ') noexcept;

template<typename T>
void printMatrix(T** matrix, int rows, int cols, const std::string& str = "");
//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE
void printfArray(T* array, int size);

template<typename T, int SIZE>
HOST_DEVICE
void printfArray(T (&array)[SIZE]) ;

/**
 * @brief left to right : char v = 1 -> 10000000
 */
template<typename T>
HOST_DEVICE void
printBits(T* array, int size);
//------------------------------------------------------------------------------

#if defined(__NVCC__)

template<class T>
void printCudaArray(const T* d_array, size_t size, const std::string& str = "",
                    char sep = ' ');

template<class T>
void printCudaSymbol(const T& d_array, size_t size, const std::string& str = "",
                     char sep = ' ');

template<class T, int SIZE>
void printCudaSymbol(const T (&d_array)[SIZE], const std::string& str = "",
                     char sep = ' ');

#endif
} // namespace xlib

#include "impl/PrintExt.i.hpp"
