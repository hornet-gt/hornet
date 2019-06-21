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
    template<typename SPtrT0, typename degree_t, typename SPtrT1>
    HOST_DEVICE
    static void assign(
            const SPtrT0& src, const degree_t srcIndex,
            SPtrT1& dst, degree_t dstIndex) {
        dst.template get<N>()[dstIndex] = src.template get<N>()[srcIndex];
        RecursiveAssign<N+1, SIZE>::assign(src, srcIndex, dst, dstIndex);
    }
    template<typename SPtr, typename degree_t, typename SRef>
    HOST_DEVICE
    static void assign(
            const SRef& src,
            SPtr& dst, degree_t dstIndex) {
        dst.template get<N>()[dstIndex] = src.template get<N>();
        RecursiveAssign<N+1, SIZE>::assign(src, dst, dstIndex);
    }
    template<typename Tuple, typename SRef>
    HOST_DEVICE
    static void assign(
            const SRef& src,
            Tuple& dst) {
        std::get<N>(dst) = src.template get<N>();
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
    template<typename SPtrT0, typename degree_t, typename SPtrT1>
    HOST_DEVICE
    static void assign(
            const SPtrT0& src, const degree_t srcIndex,
            SPtrT1& dst, degree_t dstIndex) {
        dst.template get<N>()[dstIndex] = src.template get<N>()[srcIndex];
    }
    template<typename SPtr, typename degree_t, typename SRef>
    HOST_DEVICE
    static void assign(
            const SRef& src,
            SPtr& dst, degree_t dstIndex) {
        dst.template get<N>()[dstIndex] = src.template get<N>();
    }
    template<typename Tuple, typename SRef>
    HOST_DEVICE
    static void assign(
            const SRef& src,
            Tuple& dst) {
        std::get<N>(dst) = src.template get<N>();
    }
};

//==============================================================================
///////////////////
// RecursiveCopy //
///////////////////

struct DeviceCopy {
    template<typename T>
    static void copy(
            T const * const src,
            const DeviceType src_device_type,
            T * const dst,
            const DeviceType dst_device_type,
            const int num_items) {
        if (src == nullptr) { return; }
        if ((src_device_type == DeviceType::DEVICE) &&
                (dst_device_type == DeviceType::DEVICE)) {
            cuMemcpyDevToDev(src, num_items, dst);
        } else if ((src_device_type == DeviceType::DEVICE) &&
                (dst_device_type == DeviceType::HOST)) {
            cuMemcpyToHost(src, num_items, dst);
        } else if ((src_device_type == DeviceType::HOST) &&
                (dst_device_type == DeviceType::DEVICE)) {
            cuMemcpyToDevice(src, num_items, dst);
        } else if ((src_device_type == DeviceType::HOST) &&
                (dst_device_type == DeviceType::HOST)) {
            std::copy(src, src + num_items, dst);
        }
    }
};

template<int N, int SIZE>
struct RecursiveCopy {
    template<
        template <typename...> typename SrcContnr,
        template <typename...> typename DstContnr,
        typename... Ts>
    static void copy(
            const SrcContnr<Ts...>& src,
            const DeviceType src_device_type,
            DstContnr<Ts...>& dst,
            const DeviceType dst_device_type,
            const int num_items,
            const int srcOffset = 0,
            const int dstOffset = 0) {
        DeviceCopy::copy(
                src.template get<N>() + srcOffset, src_device_type,
                dst.template get<N>() + dstOffset, dst_device_type,
                num_items);
        RecursiveCopy<N + 1, SIZE>::copy(src, src_device_type, dst, dst_device_type, num_items,
            srcOffset, dstOffset);
    }

    template<
        template <typename...> typename SrcContnr,
        template <typename...> typename DstContnr,
        typename... Ts>
    static void copy(
            const SrcContnr<Ts const...>& src,
            const DeviceType src_device_type,
            DstContnr<Ts...>& dst,
            const DeviceType dst_device_type,
            const int num_items,
            const int srcOffset = 0,
            const int dstOffset = 0) {
        DeviceCopy::copy(
                src.template get<N>() + srcOffset, src_device_type,
                dst.template get<N>() + dstOffset, dst_device_type,
                num_items);
        RecursiveCopy<N + 1, SIZE>::copy(src, src_device_type, dst, dst_device_type, num_items,
            srcOffset, dstOffset);
    }
};

template<int N>
struct RecursiveCopy<N, N> {
    template<
        template <typename...> typename SrcContnr,
        template <typename...> typename DstContnr,
        typename... Ts>
    static void copy(
            const SrcContnr<Ts...>& src,
            DeviceType src_device_type,
            DstContnr<Ts...>& dst,
            DeviceType dst_device_type,
            const int num_items,
            const int srcOffset = 0,
            const int dstOffset = 0) {
        DeviceCopy::copy(
                src.template get<N>() + srcOffset, src_device_type,
                dst.template get<N>() + dstOffset, dst_device_type,
                num_items);
    }

    template<
        template <typename...> typename SrcContnr,
        template <typename...> typename DstContnr,
        typename... Ts>
    static void copy(
            const SrcContnr<Ts const...>& src,
            DeviceType src_device_type,
            DstContnr<Ts...>& dst,
            DeviceType dst_device_type,
            const int num_items,
            const int srcOffset = 0,
            const int dstOffset = 0) {
        DeviceCopy::copy(
                src.template get<N>() + srcOffset, src_device_type,
                dst.template get<N>() + dstOffset, dst_device_type,
                num_items);
    }
};

//==============================================================================
/////////////////////
// RecursiveGather //
/////////////////////

template<unsigned N, unsigned SIZE>
struct RecursiveGather {
    template<typename degree_t, typename Ptr>
    HOST_DEVICE
    static void assign(Ptr src, Ptr dst,
        const thrust::host_vector<degree_t>& map,
        const degree_t nE) {
        if (N >= SIZE) { return; }
        thrust::gather(
                thrust::host,
                map.begin(), map.begin() + nE,
                src.template get<N>(),
                dst.template get<N>());
        RecursiveGather<N+1, SIZE>::assign(src, dst, map, nE);
    }
    template<typename degree_t, typename Ptr>
    HOST_DEVICE
    static void assign(Ptr src, Ptr dst,
        const thrust::device_vector<degree_t>& map,
        const degree_t nE) {
        if (N >= SIZE) { return; }
        thrust::gather(
                thrust::device,
                map.begin(), map.begin() + nE,
                src.template get<N>(),
                dst.template get<N>());
        RecursiveGather<N+1, SIZE>::assign(src, dst, map, nE);
    }
};

template<unsigned N>
struct RecursiveGather<N, N> {
    template<typename degree_t, typename Ptr>
    HOST_DEVICE
    static void assign(Ptr src, Ptr dst,
        const thrust::host_vector<degree_t>& map,
        const degree_t nE) { }
    template<typename degree_t, typename Ptr>
    HOST_DEVICE
    static void assign(Ptr src, Ptr dst,
        const thrust::device_vector<degree_t>& map,
        const degree_t nE) { }
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
    return *(_soa.template get<N>() + _index);
}

template<template <typename...> typename Contnr, typename... Ts>
template <unsigned N>
HOST_DEVICE
typename xlib::SelectType<N, Ts&...>::type
SoARef<Contnr<Ts...>>::
get() const& noexcept {
    return *(_soa.template get<N>() + _index);
}

template<template <typename...> typename Contnr, typename... Ts>
HOST_DEVICE
SoARef<Contnr<Ts...>>& SoARef<Contnr<Ts...>>::operator=(const SoARef<Contnr<Ts...>>& other) noexcept {
    RecursiveAssign<0, sizeof...(Ts) - 1>::assign(other, *this);
    return *this;
}

template<template <typename...> typename Contnr, typename... Ts>
template <typename EdgeT>
HOST_DEVICE
SoARef<Contnr<Ts...>>& SoARef<Contnr<Ts...>>::operator=(const EdgeT& other) noexcept {
    RecursiveAssign<0, sizeof...(Ts) - 1>::assign(other._ptr, other._index, _soa, _index);
    return *this;
}

template<template <typename...> typename Contnr, typename... Ts>
HOST_DEVICE
SoARef<Contnr<Ts...>>::SoARef(const SoARef<Contnr<Ts...>>& other) noexcept : _soa(other._soa), _index(other._index) { }

template<template <typename...> typename Contnr, typename... Ts>
HOST_DEVICE
SoARef<Contnr<Ts...>>::SoARef(Contnr<Ts...>& soa, const int& index) noexcept : _soa(soa), _index(index) { }

template<template <typename...> typename Contnr, typename... Ts>
std::tuple<Ts...> getTuple(const SoARef<Contnr<Ts...>>& r) {
  std::tuple<Ts...> t;
  RecursiveAssign<0, sizeof...(Ts) - 1>::assign(r, t);
  return t;
}

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
    return _tail.template get<N - 1>();
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
    return _tail.template get<N - 1>();
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
    _tail.template set<N-1>(ptr);
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
template <template <typename...> typename Ptr, typename degree_t, typename... EdgeTypes>
void
sort_edges(Ptr<EdgeTypes...> ptr, const degree_t nE) {
    thrust::sort_by_key(
            thrust::device,
            ptr.template get<1>(), ptr.template get<1>() + nE,
            ptr.template get<0>());
    thrust::sort_by_key(
            thrust::device,
            ptr.template get<0>(), ptr.template get<0>() + nE,
            ptr.template get<1>());
}

template <template <typename...> typename Ptr, typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 == sizeof...(EdgeTypes)), bool>::type
sort_batch(Ptr<EdgeTypes...> in_ptr, const degree_t nE, thrust::device_vector<degree_t>& range,
        Ptr<EdgeTypes...> out_ptr) {
    sort_edges(in_ptr, nE);
    return false;
}

template <template <typename...> typename Ptr, typename degree_t, typename... EdgeTypes>
typename std::enable_if<(3 == sizeof...(EdgeTypes)), bool>::type
sort_batch(Ptr<EdgeTypes...> in_ptr, const degree_t nE, thrust::device_vector<degree_t>& range,
        Ptr<EdgeTypes...> out_ptr) {
    thrust::sort_by_key(
            thrust::device,
            in_ptr.template get<1>(), in_ptr.template get<1>() + nE,
            thrust::make_zip_iterator(thrust::make_tuple(in_ptr.template get<0>(), in_ptr.template get<2>())) );
    thrust::sort_by_key(
            thrust::device,
            in_ptr.template get<0>(), in_ptr.template get<0>() + nE,
            thrust::make_zip_iterator(thrust::make_tuple(in_ptr.template get<1>(), in_ptr.template get<2>())) );
    return false;
}

template <template <typename...> typename Ptr, typename degree_t, typename... EdgeTypes>
typename std::enable_if<(3 < sizeof...(EdgeTypes)), bool>::type
sort_batch(Ptr<EdgeTypes...> in_ptr, const degree_t nE, thrust::device_vector<degree_t>& range,
        Ptr<EdgeTypes...> out_ptr) {
    range.resize(nE);
    thrust::sequence(range.begin(), range.end());
    thrust::sort_by_key(
            thrust::device,
            in_ptr.template get<1>(), in_ptr.template get<1>() + nE,
            thrust::make_zip_iterator(thrust::make_tuple(in_ptr.template get<0>(), range.begin())) );
    thrust::sort_by_key(
            thrust::device,
            in_ptr.template get<0>(), in_ptr.template get<0>() + nE,
            thrust::make_zip_iterator(thrust::make_tuple(in_ptr.template get<1>(), range.begin())) );
    RecursiveCopy<0, 2>::copy(in_ptr, DeviceType::DEVICE, out_ptr, DeviceType::DEVICE, nE);
    RecursiveGather<2, sizeof...(EdgeTypes)>::assign(in_ptr, out_ptr, range, nE);
    return true;
}

//==============================================================================
//==============================================================================
/////////////////////
// RecursiveSetPtr //
/////////////////////
template<unsigned N, unsigned SIZE, unsigned OFFSET>
struct RecursiveSetPtr {
    template<typename PtrD, typename PtrS>
    HOST_DEVICE
    static void set(PtrD &d, PtrS &s) {
        //if (N >= SIZE) { return; }
      d.template set<N>(s.template get<OFFSET>());
      RecursiveSetPtr<N+1, SIZE, OFFSET+1>::set(d, s);
    }
};

template<unsigned N, unsigned OFFSET>
struct RecursiveSetPtr<N, N, OFFSET> {
    template<typename PtrD, typename PtrS>
    HOST_DEVICE
    static void set(PtrD &d, PtrS &s) {}
};

template <typename A, typename... B>
SoAPtr<A, B...> concat(A* a, SoAPtr<B...> b) {
  SoAPtr<A, B...> d;
  d.template set<0>(a);
  RecursiveSetPtr<0, sizeof...(B), 1>::set(d, b);
  return d;
}

template <typename... A, typename... B>
SoAPtr<A..., B...> concat(SoAPtr<A...> a, SoAPtr<B...> b) {
  SoAPtr<A..., B...> d;
  RecursiveSetPtr<0, sizeof...(A), 0>::set(d, a);
  RecursiveSetPtr<0, sizeof...(B), sizeof...(A)>::set(d, b);
  return d;
}

}//namespace hornet
