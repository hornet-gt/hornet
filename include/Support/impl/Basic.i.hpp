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
#include <cstdlib>      //size_t
#include <cxxabi.h>     //abi::__cxa_demangle
#include <memory>       //std::unique_ptr
#include <string>       //std::string
#include <type_traits>  //std::remove_reference

namespace xlib {

#if !defined(__NVCC__)

constexpr int operator"" _BIT ( unsigned long long value ) {           // NOLINT
    return static_cast<int>(value);
}
constexpr size_t operator"" _KB ( unsigned long long value ) {         // NOLINT
    return static_cast<size_t>(value) * 1024llu;
}
constexpr size_t operator"" _MB ( unsigned long long value ) {         // NOLINT
    return static_cast<size_t>(value) * 1024llu * 1024llu;
}

#endif

const size_t KB = 1024llu;
const size_t MB = 1024llu * 1024llu;
const size_t GB = 1024llu * 1024llu * 1024llu;

//------------------------------------------------------------------------------

template <class T>
std::string type_name(T) {
    return type_name<T>();
}

template <class T>
std::string type_name() {
    using TR = typename std::remove_reference<T>::type;
    std::unique_ptr<char, void(*)(void*)> own
           (abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
            std::free);
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

namespace detail {

inline void printRecursive() {}

template<typename T, typename... Ts>
void printRecursive(T x, Ts... args) {
    std::cerr << x;
    printRecursive(args...);
}

} // namespace detail

} // namespace xlib
