/**
 * @internal
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
 *
 * @file
 */
#pragma once

#include "Support/Host/PrintExt.hpp"
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

bool is_integer(const std::string& str);

//------------------------------------------------------------------------------

template<typename Enum>
class PropertyClass {
public:
    explicit PropertyClass() noexcept = default;
    explicit PropertyClass(Enum value) noexcept;

    PropertyClass operator|(const PropertyClass& obj) const noexcept;
    bool          operator&(const PropertyClass& obj) const noexcept;
protected:
   explicit PropertyClass(int value) noexcept;
private:
    const int value { 0 };
};

} // namespace xlib

#include "impl/Basic.i.hpp"
