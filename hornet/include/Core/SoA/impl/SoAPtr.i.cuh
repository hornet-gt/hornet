/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
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
namespace hornet {

//==============================================================================
/////////////////////
// RecursiveAssign //
/////////////////////

template<unsigned N, unsigned SIZE>
struct RecursiveAssign {
    template<typename SRef>
    HOST_DEVICE
    static void assign(const SRef& src, SRef& dst) {
        dst.template get<N>() = src.template get<N>();
        RecursiveAssign<N+1, SIZE>::assign(src, dst);
    }
};

template<unsigned N>
struct RecursiveAssign<N, N> {
    template<typename SRef>
    HOST_DEVICE
    static void assign(const SRef& src, SRef& dst) {
        dst.template get<N>() = src.template get<N>();
    }
};

//==============================================================================
////////////
// SoARef //
////////////

template<template <typename...> typename Contnr, typename... Ts>
template <unsigned N>
HOST_DEVICE
typename xlib::SelectType<N, Ts&...>::type
SoARef<Contnr<Ts...>>::
get() & noexcept {
    return *(_soa.get<N>() + _index);
}

template<template <typename...> typename Contnr, typename... Ts>
template <unsigned N>
HOST_DEVICE
typename xlib::SelectType<N, Ts&...>::type
SoARef<Contnr<Ts...>>::
get() const& noexcept {
    return *(_soa.get<N>() + _index);
}

template<template <typename...> typename Contnr, typename... Ts>
HOST_DEVICE
SoARef<Contnr<Ts...>>& SoARef<Contnr<Ts...>>::operator=(const SoARef<Contnr<Ts...>>& other) noexcept {
    RecursiveAssign<0, sizeof...(Ts) - 1>::assign(other, *this);
    return *this;
}

template<template <typename...> typename Contnr, typename... Ts>
HOST_DEVICE
SoARef<Contnr<Ts...>>::SoARef(const SoARef<Contnr<Ts...>>& other) noexcept : _soa(other._soa), _index(other._index) { }

template<template <typename...> typename Contnr, typename... Ts>
HOST_DEVICE
SoARef<Contnr<Ts...>>::SoARef(Contnr<Ts...>& soa, const int& index) noexcept : _soa(soa), _index(index) { }

//==============================================================================
///////////////
// SoAPtr<T> //
///////////////

template<typename T>
HOST_DEVICE
SoAPtr<T>::
SoAPtr(void) noexcept :
_ptr(nullptr) {}

template<typename T>
HOST_DEVICE
SoAPtr<T>::
SoAPtr(T* const ptr) noexcept :
_ptr(ptr) {}

template<typename T>
template<unsigned N>
HOST_DEVICE
T* SoAPtr<T>::
get() noexcept {
    static_assert(N == 0, "error");
    return _ptr;
}

template<typename T>
template<unsigned N>
HOST_DEVICE
T const * SoAPtr<T>::
get() const noexcept {
    static_assert(N == 0, "error");
    return _ptr;
}

template<typename T>
template<unsigned N>
void SoAPtr<T>::
set(T* const ptr) noexcept {
    _ptr = ptr;
}

template<typename T>
HOST_DEVICE
SoARef<SoAPtr<T>> SoAPtr<T>::
operator[](const int& index)  noexcept {
    return SoARef<SoAPtr<T>>(*this, index);
}

//==============================================================================
/////////////////////
// SoAPtr<T, Ts..> //
/////////////////////


template<typename T, typename... Ts>
HOST_DEVICE
SoAPtr<T, Ts...>::
SoAPtr(void) noexcept :
_ptr(nullptr), _tail() {}

template<typename T, typename... Ts>
HOST_DEVICE
SoAPtr<T, Ts...>::
SoAPtr(T* const ptr, Ts* const... args) noexcept :
_ptr(ptr),
_tail(args...) {}

template<typename T, typename... Ts>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<N == 0, T*>::type
SoAPtr<T, Ts...>::
get() noexcept {
    return _ptr;
}

template<typename T, typename... Ts>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<N != 0, typename xlib::SelectType<N, T*, Ts*...>::type>::type
SoAPtr<T, Ts...>::
get() noexcept {
    return _tail.get<N - 1>();
}

template<typename T, typename... Ts>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<N == 0, T const *>::type
SoAPtr<T, Ts...>::
get() const noexcept {
    return _ptr;
}

template<typename T, typename... Ts>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<N != 0, typename xlib::SelectType<N, T const *, Ts const *...>::type>::type
SoAPtr<T, Ts...>::
get() const noexcept {
    return _tail.get<N - 1>();
}

template<typename T, typename... Ts>
template<unsigned N>
void SoAPtr<T, Ts...>::
set(typename std::enable_if<N == 0, T* const>::type ptr) noexcept {
    _ptr = ptr;
}

template<typename T, typename... Ts>
template<unsigned N>
void SoAPtr<T, Ts...>::
set(typename std::enable_if<N != 0, typename xlib::SelectType<N, T* const, Ts* const...>::type>::type ptr) noexcept {
    _tail.set<N-1>(ptr);
}

template<typename T, typename... Ts>
HOST_DEVICE
SoARef<SoAPtr<T, Ts...>>
SoAPtr<T, Ts...>::
operator[](const int& index)  noexcept {
    return SoARef<SoAPtr<T, Ts...>>(*this, index);
}

template<typename T, typename... Ts>
HOST_DEVICE
SoAPtr<Ts...>
SoAPtr<T, Ts...>::
get_tail(void) noexcept {
    return _tail;
}

//==============================================================================
//////////////////////
// CSoAPtr<T, Ts..> //
//////////////////////


template<typename T, typename... Ts>
HOST_DEVICE
CSoAPtr<T, Ts...>::
CSoAPtr(void) noexcept :
_ptr(nullptr), _num_items(0) {}

template<typename T, typename... Ts>
HOST_DEVICE
CSoAPtr<T, Ts...>::
CSoAPtr(xlib::byte_t* const ptr, const int num_items) noexcept :
_ptr(ptr),
_num_items(num_items) {}

template<typename T, typename... Ts>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<(N < (1 + sizeof...(Ts))),
         typename xlib::SelectType<N, T*, Ts*...>::type
         >::type
CSoAPtr<T, Ts...>::
get() noexcept {
    return reinterpret_cast<typename xlib::SelectType<N, T*, Ts*...>::type>(
            _ptr + xlib::FirstNSizeSum<N, T, Ts...>::value * _num_items);
}

template<typename T, typename... Ts>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<(N < (1 + sizeof...(Ts))),
         typename xlib::SelectType<N, T const *, Ts const *...>::type
         >::type
CSoAPtr<T, Ts...>::
get() const noexcept {
    return reinterpret_cast<typename xlib::SelectType<N, T const *, Ts const *...>::type>(
            _ptr + xlib::FirstNSizeSum<N, T, Ts...>::value * _num_items);
}

template<typename T, typename... Ts>
HOST_DEVICE
SoARef<CSoAPtr<T, Ts...>>
CSoAPtr<T, Ts...>::
operator[](const int& index)  noexcept {
    return SoARef<CSoAPtr<T, Ts...>>(*this, index);
}

template<typename T, typename... Ts>
HOST_DEVICE
CSoAPtr<Ts...>
CSoAPtr<T, Ts...>::
get_tail(void) noexcept {
    return CSoAPtr<Ts...>(reinterpret_cast<xlib::byte_t*>(get<1>()), _num_items);
}

//==============================================================================
//////////////////////
// CSoAPtr<T> //
//////////////////////


template<typename T>
HOST_DEVICE
CSoAPtr<T>::
CSoAPtr(void) noexcept :
_ptr(nullptr), _num_items(0) {}

template<typename T>
HOST_DEVICE
CSoAPtr<T>::
CSoAPtr(xlib::byte_t* const ptr, const int num_items) noexcept :
_ptr(ptr),
_num_items(num_items) {}

template<typename T>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<(N == 0), T*>::type
CSoAPtr<T>::
get() noexcept {
    return reinterpret_cast<T*>(_ptr);
}

template<typename T>
template<unsigned N>
HOST_DEVICE
typename std::enable_if<(N == 0), T const *>::type
CSoAPtr<T>::
get() const noexcept {
    return reinterpret_cast<T const *>(_ptr);
}

template<typename T>
HOST_DEVICE
SoARef<CSoAPtr<T>>
CSoAPtr<T>::
operator[](const int& index)  noexcept {
    return SoARef<CSoAPtr<T>>(*this, index);
}

//==============================================================================
//==============================================================================

}//namespace hornet
