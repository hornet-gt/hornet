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
//////////////////////////////////////
// Device Specialized De/Allocation //
//////////////////////////////////////

//Avoiding specialization for hornetsnest
template<DeviceType device_t>
xlib::byte_t* allocate(const typename std::enable_if<(device_t == DeviceType::DEVICE), int>::type
        num_bytes) {
    xlib::byte_t* ptr = nullptr;
    if (num_bytes > 0) {
        cuMalloc(ptr, num_bytes);
    }
    return ptr;
}

template<DeviceType device_t>
void deallocate(typename std::enable_if<(device_t == DeviceType::DEVICE), xlib::byte_t>::type *
        ptr) {
    if (ptr != nullptr) {
        cuFree(ptr);
    }
}

template<DeviceType device_t>
xlib::byte_t* allocate(const typename std::enable_if<(device_t == DeviceType::HOST), int>::type
        num_bytes) {
    xlib::byte_t* ptr = nullptr;
    if (num_bytes > 0) {
        ptr = new xlib::byte_t[num_bytes];
    }
    return ptr;
}

template<DeviceType device_t>
void deallocate(typename std::enable_if<(device_t == DeviceType::HOST), xlib::byte_t>::type *
 ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
    }
}

//==============================================================================
////////////////////
// AllocateBuffer //
////////////////////

template<int N, DeviceType device_t>
struct AllocateBuffer {
    template<template <typename...> typename Contnr, typename... Ts>
    static void allocate(Contnr<Ts...>& c, const int num_items) {
        typename xlib::SelectType<N, Ts*...>::type ptr;
        cuMalloc(ptr, num_items);
        c.template set<N>(ptr);
    }
};

template<int N>
struct AllocateBuffer<N, DeviceType::HOST> {
    template<template <typename...> typename Contnr, typename... Ts>
    static void allocate(Contnr<Ts...>& c, const int num_items) {
        using val_t = typename xlib::SelectType<N, Ts...>::type;
        val_t* ptr = new val_t[num_items];
        c.template set<N>(ptr);
    }
};

//==============================================================================
//////////////////////
// DeallocateBuffer //
//////////////////////

template<int N, DeviceType device_t>
struct DeallocateBuffer {
    template<template <typename...> typename Contnr, typename... Ts>
    static void deallocate(Contnr<Ts...>& c) {
        typename xlib::SelectType<N, Ts*...>::type ptr = c.template get<N>();
        cuFree(ptr);
        c.template set<N>(nullptr);
    }
};

template<int N>
struct DeallocateBuffer<N, DeviceType::HOST> {
    template<template <typename...> typename Contnr, typename... Ts>
    static void deallocate(Contnr<Ts...>& c) {
        delete[] c.template get<N>();
        c.template set<N>(nullptr);
    }
};

//==============================================================================
///////////////////
// RecursiveMove //
///////////////////

template<int N, int SIZE, DeviceType device_t>
struct RecursiveMove {
    template<template <typename...> typename Contnr, typename... Ts>
    static void move(Contnr<Ts...>& c_src, Contnr<Ts...>& c_dst) {
        c_dst.template set<N>(c_src.template get<N>());
        c_src.template set<N>(nullptr);
        RecursiveMove<N+1, SIZE, device_t>::move(c_src, c_dst);
    }
};

template<int N, DeviceType device_t>
struct RecursiveMove<N, N, device_t> {
    template<template <typename...> typename Contnr, typename... Ts>
    static void move(Contnr<Ts...>& c_src, Contnr<Ts...>& c_dst) {
        c_dst.template set<N>(c_src.template get<N>());
        c_src.template set<N>(nullptr);
    }
};

//==============================================================================
///////////////////////
// RecursiveAllocate //
///////////////////////

template<int N, int SIZE, DeviceType device_t>
struct RecursiveAllocate {
    template<template <typename...> typename Contnr, typename... Ts>
    static void allocate(Contnr<Ts...>& c, const int num_items) {
        AllocateBuffer<N, device_t>::allocate(c, num_items);
        RecursiveAllocate<N+1, SIZE, device_t>::allocate(c, num_items);
    }
};

template<int N, DeviceType device_t>
struct RecursiveAllocate<N, N, device_t> {
    template<template <typename...> typename Contnr, typename... Ts>
    static void allocate(Contnr<Ts...>& c, const int num_items) {
        AllocateBuffer<N, device_t>::allocate(c, num_items);
    }
};

//==============================================================================
/////////////////////////
// RecursiveDeallocate //
/////////////////////////

template<int N, int SIZE, DeviceType device_t>
struct RecursiveDeallocate {
    template<template <typename...> typename Contnr, typename... Ts>
    static void deallocate(Contnr<Ts...>& c) {
        DeallocateBuffer<N, device_t>::deallocate(c);
        RecursiveDeallocate<N+1, SIZE, device_t>::deallocate(c);
    }
};

template<int N, DeviceType device_t>
struct RecursiveDeallocate<N, N, device_t> {
    template<template <typename...> typename Contnr, typename... Ts>
    static void deallocate(Contnr<Ts...>& c) {
        DeallocateBuffer<N, device_t>::deallocate(c);
    }
};

//==============================================================================
//////////////////////
// RecursiveSetNull //
//////////////////////

template<int N, int SIZE>
struct RecursiveSetNull {
    template<template <typename...> typename Contnr, typename... Ts>
    static void set_null(Contnr<Ts...>& c) {
        c.template set<N>(nullptr);
        RecursiveSetNull<N+1, SIZE>::set_null(c);
    }
};

template<int N>
struct RecursiveSetNull<N, N> {
    template<template <typename...> typename Contnr, typename... Ts>
    static void set_null(Contnr<Ts...>& c) {
        c.template set<N>(nullptr);
    }
};

//==============================================================================
///////////////////
// RecursivePrint //
///////////////////

struct DevicePrint {
    template<typename T>
    static void print(
            T * src,
            DeviceType src_device_type,
            int num_items) {
        if (src_device_type == DeviceType::DEVICE) {
          std::cout<<"TODO : Implement\n";
        } else if (src_device_type == DeviceType::HOST) {
            for (int i = 0; i < num_items; ++i) {
                std::cout<<src[i]<<" ";
            }
        }
    }
};

template<int N, int SIZE>
struct RecursivePrint {
    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
        RecursivePrint<N + 1, SIZE>::print(src, src_device_type, num_items);
    }

    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts const...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
        RecursivePrint<N + 1, SIZE>::print(src, src_device_type, num_items);
    }
};

template<int N>
struct RecursivePrint<N, N> {
    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
    }

    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts const...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
    }
};

//==============================================================================
/////////////
// SoAData //
/////////////

template<typename... Ts, DeviceType device_t>
SoAData<TypeList<Ts...>, device_t>::
SoAData(const int num_items) noexcept :
_num_items(num_items), _capacity(num_items) {
    if (num_items != 0) {
        RecursiveAllocate<0, sizeof...(Ts) - 1, device_t>::allocate(_soa, _capacity);
    }
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
SoAData<TypeList<Ts...>, device_t>::
SoAData(SoAData<TypeList<Ts...>, d_t>&& other) noexcept :
_num_items(other._num_items), _capacity(other._capacity) {
    if (d_t != device_t) {
        //If other's device type is different, do not alter its variables
        RecursiveAllocate<0, sizeof...(Ts) - 1, device_t>::allocate(_soa, _capacity);
        RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, d_t, _soa, device_t, _num_items);
    } else {
        RecursiveMove<0, sizeof...(Ts) - 1, device_t>::move(other._soa, _soa);
        RecursiveSetNull<0, sizeof...(Ts) - 1>::set_null(other._soa);
        other._num_items = 0;
        other._capacity = 0;
    }
}

template<typename... Ts, DeviceType device_t>
SoAData<TypeList<Ts...>, device_t>::
~SoAData(void) noexcept {
    RecursiveDeallocate<0, sizeof...(Ts) - 1, device_t>::deallocate(_soa);
}

template<typename... Ts, DeviceType device_t>
SoAData<TypeList<Ts...>, device_t>&
SoAData<TypeList<Ts...>, device_t>::
operator=(SoAData<TypeList<Ts...>, device_t>&& other) {
    RecursiveDeallocate<0, sizeof...(Ts) - 1, device_t>::deallocate(_soa);
    RecursiveMove<0, sizeof...(Ts) - 1, device_t>::move(other._soa, _soa);
    RecursiveSetNull<0, sizeof...(Ts) - 1>::set_null(other._soa);
    _num_items = other._num_items;
    _capacity = other._capacity;
    other._num_items = 0;
    other._capacity = 0;
    return *this;
}

template<typename... Ts, DeviceType device_t>
SoAPtr<Ts...>&
SoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
const SoAPtr<Ts...>&
SoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) const noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
void
SoAData<TypeList<Ts...>, device_t>:://TODO remove get_soa_ptr because SoAData<typename, DeviceType> is friend
copy(const SoAData<TypeList<Ts...>, d_t>& other) noexcept {
    int _item_count = std::min(other._num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
void
SoAData<TypeList<Ts...>, device_t>:://TODO remove get_soa_ptr because SoAData<typename, DeviceType> is friend
copy(SoAPtr<Ts...> other, const DeviceType other_d_t, const int other_num_items) noexcept {
    int _item_count = std::min(other_num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, other_d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
void
SoAData<TypeList<Ts...>, device_t>:://TODO remove get_soa_ptr because SoAData<typename, DeviceType> is friend
append(const SoAData<TypeList<Ts...>, d_t>& other) noexcept {
  auto old_num_items = _num_items;
  resize(_num_items + other._num_items);
  RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, d_t, _soa, device_t, other._num_items, 0, old_num_items);
}

template<typename... Ts, DeviceType device_t>
void
SoAData<TypeList<Ts...>, device_t>::
sort(void) noexcept {
  thrust::device_vector<int> range;
  if (sizeof...(Ts) > 3) {
    SoAPtr<Ts...> temp_soa;
    RecursiveAllocate<0, sizeof...(Ts) - 1, device_t>::allocate(temp_soa, _num_items);
    sort_batch(_soa, _num_items, range, temp_soa);
    RecursiveDeallocate<0, sizeof...(Ts) - 1, device_t>::deallocate(_soa);
    _soa = temp_soa;
  } else {
    SoAPtr<Ts...> temp_soa;
    sort_batch(_soa, _num_items, range, temp_soa);
  }
}

template<typename... Ts, DeviceType device_t>
template <typename degree_t>
void
SoAData<TypeList<Ts...>, device_t>::
gather(SoAData<TypeList<Ts...>, device_t>& other, const Map<degree_t>& map) noexcept {
  RecursiveGather<0, sizeof...(Ts)>::assign(other._soa, _soa, map, static_cast<degree_t>(map.size()));
}

template<typename... Ts, DeviceType device_t>
int
SoAData<TypeList<Ts...>, device_t>::
get_num_items(void) noexcept {
    return _num_items;
}

template<typename... Ts, DeviceType device_t>
void
SoAData<TypeList<Ts...>, device_t>::
resize(const int resize_items) noexcept {
    if (resize_items > _capacity) {
        SoAPtr<Ts...> temp_soa;
        RecursiveAllocate<0, sizeof...(Ts) - 1, device_t>::allocate(temp_soa, resize_items);
        RecursiveCopy<0, sizeof...(Ts) - 1>::copy(_soa, device_t, temp_soa, device_t, _num_items);
        RecursiveDeallocate<0, sizeof...(Ts) - 1, device_t>::deallocate(_soa);
        _soa = temp_soa;
        _capacity = resize_items;
    }
    _num_items = resize_items;
}

template<typename... Ts, DeviceType device_t>
DeviceType
SoAData<TypeList<Ts...>, device_t>::
get_device_type(void) noexcept {
    return device_t;
}

//==============================================================================
//////////////
// CSoAData //
//////////////

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>::
CSoAData(const int num_items) noexcept :
_num_items(num_items), _capacity(xlib::upper_approx<512>(num_items)),
_soa(allocate<device_t>(xlib::SizeSum<Ts...>::value * _capacity), _capacity) {}

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>::
CSoAData(CSoAData<TypeList<Ts...>, device_t>&& other) noexcept :
_num_items(other._num_items), _capacity(xlib::upper_approx<512>(_num_items)),
_soa(nullptr, 0) {
    _soa = other._soa;
    other._soa = CSoAPtr<Ts...>(nullptr, 0);
    other._num_items = 0;
    other._capacity = 0;
}

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>&
CSoAData<TypeList<Ts...>, device_t>::
operator=(CSoAData<TypeList<Ts...>, device_t>&& other) {
    deallocate<device_t>(reinterpret_cast<xlib::byte_t*>(_soa.template get<0>()));
    _soa = other._soa;
    other._soa = CSoAPtr<Ts...>(nullptr, 0);
    other._num_items = 0;
    other._capacity = 0;
    return *this;
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
CSoAData<TypeList<Ts...>, device_t>::
CSoAData(CSoAData<TypeList<Ts...>, d_t>&& other) noexcept :
_num_items(other._num_items), _capacity(other._capacity),
_soa(nullptr, 0) {
    if (d_t != device_t) {
        //If other's device type is different, do not alter its variables
        _soa =
            CSoAPtr<Ts...>(
                    allocate<device_t>(xlib::SizeSum<Ts...>::value * _capacity),
                    _capacity);
        copy(other);
    } else {
        _soa = other._soa;
        other._soa = CSoAPtr<Ts...>(nullptr, 0);
        other._num_items = 0;
        other._capacity = 0;
    }
}

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>::
~CSoAData(void) noexcept {
    deallocate<device_t>(reinterpret_cast<xlib::byte_t*>(_soa.template get<0>()));
}

template<typename... Ts, DeviceType device_t>
CSoAPtr<Ts...>&
CSoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
const CSoAPtr<Ts...>&
CSoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) const noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
void
CSoAData<TypeList<Ts...>, device_t>::
copy(SoAPtr<Ts...> other, const DeviceType other_d_t, const int other_num_items) noexcept {
    int _item_count = std::min(other_num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other, other_d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
void
CSoAData<TypeList<Ts...>, device_t>::
copy(CSoAData<TypeList<Ts...>, d_t>&& other) noexcept {
    //Copy the whole block as is
    if ((other._capacity == _capacity) &&
            (other._capacity == other._num_items) &&
            (other._num_items == _num_items)) {
        DeviceCopy::copy(
                reinterpret_cast<xlib::byte_t const *>(
                    other.get_soa_ptr().template get<0>()), d_t,
                reinterpret_cast<xlib::byte_t*>(
                    _soa.template get<0>()), device_t,
                xlib::SizeSum<Ts...>::value * other._capacity);
    } else {
        int _item_count = std::min(other._num_items, _num_items);
        RecursiveCopy<0, sizeof...(Ts) - 1>::copy(
                other.get_soa_ptr(), d_t, _soa, device_t, _item_count);
    }
}

template<typename... Ts, DeviceType device_t>
int
CSoAData<TypeList<Ts...>, device_t>::
get_num_items(void) noexcept {
    return _num_items;
}

template<typename... Ts, DeviceType device_t>
void
CSoAData<TypeList<Ts...>, device_t>::
resize(const int resize_items) noexcept {
    int new_capacity = xlib::upper_approx<512>(resize_items);
    if (new_capacity > _capacity) {
        CSoAPtr<Ts...> temp_soa(
                allocate<device_t>(xlib::SizeSum<Ts...>::value * new_capacity),
                new_capacity);
        RecursiveCopy<0, sizeof...(Ts) - 1>::copy(
                _soa, device_t, temp_soa, device_t, _num_items);
        deallocate<device_t>(reinterpret_cast<xlib::byte_t*>(_soa.template get<0>()));
        _soa = temp_soa;
        _capacity = new_capacity;
    }
    _num_items = resize_items;
}

template<typename... Ts, DeviceType device_t>
DeviceType
CSoAData<TypeList<Ts...>, device_t>::
get_device_type(void) noexcept {
    return device_t;
}

//==============================================================================
//print data


template<typename... Ts>
void print(SoAData<TypeList<Ts...>, DeviceType::HOST>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::HOST, data.get_num_items());
}

template<typename... Ts>
void print(SoAData<TypeList<Ts...>, DeviceType::DEVICE>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::DEVICE, data.get_num_items());
}

template<typename... Ts>
void print(CSoAData<TypeList<Ts...>, DeviceType::HOST>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::HOST, data.get_num_items());
}

template<typename... Ts>
void print(CSoAData<TypeList<Ts...>, DeviceType::DEVICE>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::DEVICE, data.get_num_items());
}

//==============================================================================

}
