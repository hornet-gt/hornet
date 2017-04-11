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

#include "Support/PrintExt.hpp"
#include <cassert>                  //assert
#include <cstdlib>                  //std::exit
#include <iostream>                 //std::cout
#include <string>                   //std::string

#if !defined(__NVCC__)

#define ERROR(...) {                                                           \
    std::cerr << xlib::Color::FG_RED << "\nHOST error\n"                       \
              << xlib::Color::FG_DEFAULT;                                      \
    xlib::detail::printRecursive(__VA_ARGS__);                                 \
    std::cerr << "\n" << std::endl;                                            \
    assert(false);                                                  /*NOLINT*/ \
    std::exit(EXIT_FAILURE);                                                   \
}

#define ERROR_LINE {                                                           \
    std::cerr << xlib::Color::FG_RED << "\nHOST error\n"                       \
              << xlib::Color::FG_DEFAULT << xlib::Emph::SET_UNDERLINE          \
              << __FILE__ << xlib::Emph::SET_RESET  << "(" << __LINE__ << ")"  \
              << " [ " << xlib::Color::FG_L_CYAN << __func__                   \
              << xlib::Color::FG_DEFAULT << " ]\n" << std::endl;               \
    assert(false);                                                 /*NOLINT*/  \
    std::exit(EXIT_FAILURE);                                                   \
}

#else

#define ERROR(...) {                                                           \
    std::cerr << xlib::Color::FG_RED << "\nHOST error\n"                       \
              << xlib::Color::FG_DEFAULT;                                      \
    xlib::detail::printRecursive(__VA_ARGS__);                                 \
    std::cerr << "\n" << std::endl;                                            \
    assert(false);                                                  /*NOLINT*/ \
    std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));                 \
    std::exit(EXIT_FAILURE);                                                   \
}

#define ERROR_LINE {                                                           \
    std::cerr << xlib::Color::FG_RED << "\nHOST error\n"                       \
              << xlib::Color::FG_DEFAULT << xlib::Emph::SET_UNDERLINE          \
              << __FILE__ << xlib::Emph::SET_RESET  << "(" << __LINE__ << ")"  \
              << " [ " << xlib::Color::FG_L_CYAN << __func__                   \
              << xlib::Color::FG_DEFAULT << " ]\n" << std::endl;               \
    assert(false);                                                  /*NOLINT*/ \
    std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));                 \
    std::exit(EXIT_FAILURE);                                                   \
}

#endif

#define WARNING(...) {                                                         \
    std::cerr << xlib::Color::FG_L_YELLOW << "\nWarning\t"                     \
              << xlib::Color::FG_DEFAULT;                                      \
    xlib::detail::printRecursive(__VA_ARGS__);                                 \
    std::cerr << "\n" << std::endl;                                            \
}

///Computes the offset of the field in the structure/class
#define FIELD_OFFSET(structure, field)                                         \
                         (&((reinterpret_cast<structure*>(0))->field))  //NOLINT

//==============================================================================

#if defined(__CYGWIN__)
namespace std {
    template <typename T>
    std::string to_string(const T& value) {
        std::ostringstream stm;
        stm << value;
        return stm.str();
    }
} // namespace std
#endif

//==============================================================================

namespace xlib {

using byte_t = uint8_t;

#if !defined(__NVCC__)

constexpr int    operator"" _BIT ( long long unsigned value );         // NOLINT
constexpr size_t operator"" _KB ( long long unsigned value );          // NOLINT
constexpr size_t operator"" _MB ( long long unsigned value );          // NOLINT

#endif

template<typename>
struct get_arity;

template<typename R, typename... Args>
struct get_arity<R(*)(Args...)> {
    static const int value = sizeof...(Args);
};

template<typename R, typename C, typename... Args>
struct get_arity<R(C::*)(Args...)> {
    static const int value = sizeof...(Args);
};

template <class T>
std::string type_name(T Obj);

//@see http://stackoverflow.com/posts/20170989/revisions
template <class T>
std::string type_name();

//------------------------------------------------------------------------------

void memInfoHost(size_t request) noexcept;

void ctrlC_Handle();

namespace detail {
    void memInfoPrint(size_t total, size_t free, size_t request) noexcept;
} // namespace detail

} // namespace xlib

#include "impl/Basic.i.hpp"
