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
#ifndef SOAPTR_CUH
#define SOAPTR_CUH

#include "Host/Metaprogramming.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <Device/Util/SafeCudaAPI.cuh>  //cuMalloc
#include <Device/Util/SafeCudaAPISync.cuh>
//#include "BasicTypes.hpp"   //xlib::byte_t

namespace hornet {

//==============================================================================
////////////
// SoARef //
////////////

template<typename T> class SoARef;

template<template <typename...> typename Contnr, typename... Ts>
class SoARef<Contnr<Ts...>> {

    friend class Contnr<Ts...>;
    template <unsigned, unsigned> friend class RecursiveAssign;

public:

    template <unsigned N>
    HOST_DEVICE
    typename xlib::SelectType<N, Ts&...>::type
    get() & noexcept;

    template <unsigned N>
    HOST_DEVICE
    typename xlib::SelectType<N, Ts&...>::type
    get() const& noexcept;

    HOST_DEVICE
    SoARef<Contnr<Ts...>>& operator=(const SoARef<Contnr<Ts...>>& other) noexcept;

    template <typename EdgeT>
    HOST_DEVICE
    SoARef<Contnr<Ts...>>& operator=(const EdgeT& other) noexcept;

    HOST_DEVICE
    SoARef(const SoARef<Contnr<Ts...>>& other) noexcept;

    HOST_DEVICE
    SoARef(Contnr<Ts...>& soa, const int& index) noexcept;

private:

    Contnr<Ts...>&  _soa;
    const int     _index;
};

//==============================================================================
////////////
// SoAPtr //
////////////

template<typename... Ts>
class SoAPtr;

template<typename T>
class SoAPtr<T> {
public:
    HOST_DEVICE
    explicit SoAPtr() noexcept;

    HOST_DEVICE
    explicit SoAPtr(T* const ptr) noexcept;

    template<unsigned N = 0>
    HOST_DEVICE
    T*
    get() noexcept;

    template<unsigned N = 0>
    HOST_DEVICE
    T const *
    get() const noexcept;

    template<unsigned N>
    void set(T* const ptr) noexcept;

    HOST_DEVICE
    SoARef<SoAPtr<T>> operator[](const int& index) noexcept;

private:
    T* _ptr;
};

template<typename T, typename... Ts>
class
SoAPtr<T, Ts...> {
public:
    HOST_DEVICE
    explicit SoAPtr() noexcept;

    HOST_DEVICE
    explicit SoAPtr(T* const ptr, Ts* const... args) noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<N == 0, T*>::type
    get() noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<N != 0, typename xlib::SelectType<N, T*, Ts*...>::type>::type
    get() noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<N == 0, T const *>::type
    get() const noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<N != 0, typename xlib::SelectType<N, T const *, Ts const *...>::type>::type
    get() const noexcept;

    template<unsigned N>
    void set(typename std::enable_if<N == 0, T* const>::type ptr) noexcept;

    template<unsigned N>
    void set(typename std::enable_if<N != 0, typename xlib::SelectType<N, T* const, Ts* const...>::type>::type ptr) noexcept;

    HOST_DEVICE
    SoARef<SoAPtr<T, Ts...>> operator[](const int& index) noexcept;

    HOST_DEVICE
    SoAPtr<Ts...> get_tail(void) noexcept;

private:
    T*             _ptr;
    SoAPtr<Ts...> _tail;
};

//==============================================================================
/////////////
// CSoAPtr //
/////////////

template<typename... Ts>
class CSoAPtr;

template<typename T>
class
CSoAPtr<T> {
public:
    HOST_DEVICE
    explicit CSoAPtr() noexcept;

    HOST_DEVICE
    explicit CSoAPtr(xlib::byte_t* const ptr, const int num_items) noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<(N == 0), T*>::type
    get() noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<(N == 0), T const *>::type
    get() const noexcept;

    HOST_DEVICE
    SoARef<CSoAPtr<T>> operator[](const int& index) noexcept;

private:
    xlib::byte_t* _ptr;
    int           _num_items;
};

template<typename T, typename... Ts>
class
CSoAPtr<T, Ts...> {
public:
    HOST_DEVICE
    explicit CSoAPtr() noexcept;

    HOST_DEVICE
    explicit CSoAPtr(xlib::byte_t* const ptr, const int num_items) noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<(N < (1 + sizeof...(Ts))), typename xlib::SelectType<N, T*, Ts*...>::type>::type
    get() noexcept;

    template<unsigned N>
    HOST_DEVICE
    typename std::enable_if<(N < (1 + sizeof...(Ts))), typename xlib::SelectType<N, T const*, Ts const*...>::type>::type
    get() const noexcept;

    HOST_DEVICE
    SoARef<CSoAPtr<T, Ts...>> operator[](const int& index) noexcept;

    HOST_DEVICE
    CSoAPtr<Ts...> get_tail(void) noexcept;

private:
    xlib::byte_t* _ptr;
    int           _num_items;
};

}//namespace hornet

#include "impl/SoAPtr.i.cuh"
#endif
