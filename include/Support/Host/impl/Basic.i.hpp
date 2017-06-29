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

inline bool is_integer(const std::string& str) {
    return str.find_first_not_of("0123456789") == std::string::npos;
}

//------------------------------------------------------------------------------

template<typename Enum, typename CRTP>
PropertyClass<Enum, CRTP>::PropertyClass(const Enum& value) noexcept :
                        _state(static_cast<int>(value)) {}

template<typename Enum, typename CRTP>
CRTP PropertyClass<Enum, CRTP>::operator| (const CRTP& obj) const noexcept {
    CRTP crtp;
    crtp._state = _state | obj._state;
    return crtp;
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::operator& (const CRTP& obj)
                                     const noexcept {
    return _state & obj._state;
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::operator== (const CRTP& obj)
                                      const noexcept {
    return _state & obj._state;
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::is_undefined() const noexcept {
    return _state == 0;
}

template<typename Enum, typename CRTP>
void PropertyClass<Enum, CRTP>::operator+= (const CRTP& obj) noexcept {
    _state |= obj._state;
}

template<typename Enum, typename CRTP>
void PropertyClass<Enum, CRTP>::operator-= (const CRTP& obj) noexcept {
    _state &= ~obj._state;
}

template<typename Enum, typename CRTP>
CRTP& PropertyClass<Enum, CRTP>::operator= (const CRTP& obj) noexcept {
    _state = obj._state;
    return *this;
}

} // namespace xlib
