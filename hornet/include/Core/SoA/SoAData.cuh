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
#ifndef SOADATA_CUH
#define SOADATA_CUH

#include "../Conf/Common.cuh"
#include "SoAPtr.cuh"
#include <Device/Util/SafeCudaAPI.cuh>  //cuMalloc
#include <Device/Util/SafeCudaAPISync.cuh>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <vector>

namespace hornet {

template <typename, DeviceType = DeviceType::DEVICE> class SoAData;
template <typename, DeviceType = DeviceType::DEVICE> class CSoAData;

template<typename... Ts, DeviceType device_t>
class SoAData<TypeList<Ts...>, device_t> {
    template<typename, DeviceType> friend class SoAData;
    int           _num_items;

    int            _capacity;

    SoAPtr<Ts...> _soa;

    public:
    template <typename T>
    using Map = typename
    std::conditional<
    (device_t == DeviceType::DEVICE),
    typename thrust::device_vector<T>,
    typename thrust::host_vector<T>>::type;

    SoAData(const int num_items = 0) noexcept;

    ~SoAData(void) noexcept;

    SoAData& operator=(const SoAData&) = delete;

    SoAData& operator=(SoAData&& other);

    template<DeviceType d_t>
    SoAData(SoAData<TypeList<Ts...>, d_t>&& other) noexcept;

    SoAPtr<Ts...>& get_soa_ptr(void) noexcept;

    const SoAPtr<Ts...>& get_soa_ptr(void) const noexcept;

    template<DeviceType d_t>
    void copy(const SoAData<TypeList<Ts...>, d_t>& other) noexcept;

    void copy(SoAPtr<Ts...> other, const DeviceType other_d_t, const int other_num_items) noexcept;

    template<DeviceType d_t>
    void append(const SoAData<TypeList<Ts...>, d_t>& other) noexcept;

    void sort(void) noexcept;

    template <typename degree_t>
    void gather(SoAData<TypeList<Ts...>, device_t>& other, const Map<degree_t>& map) noexcept;

    int get_num_items(void) noexcept;

    void resize(const int resize_items) noexcept;

    DeviceType get_device_type(void) noexcept;
};

template<typename... Ts, DeviceType device_t>
class CSoAData<TypeList<Ts...>, device_t> {
    template<typename, DeviceType> friend class CSoAData;
    int            _num_items;

    int             _capacity;

    CSoAPtr<Ts...> _soa;

    public:
    CSoAData(const int num_items = 0) noexcept;

    ~CSoAData(void) noexcept;

    CSoAData& operator=(const CSoAData&) = delete;

    CSoAData& operator=(CSoAData&&);

    //CSoAData(const CSoAData<TypeList<Ts...>, device_t>& other) noexcept;

    CSoAData(CSoAData<TypeList<Ts...>, device_t>&& other) noexcept;

    //template<DeviceType d_t>
    //CSoAData(const CSoAData<TypeList<Ts...>, d_t>& other) noexcept;

    template<DeviceType d_t>
    CSoAData(CSoAData<TypeList<Ts...>, d_t>&& other) noexcept;

    CSoAPtr<Ts...>& get_soa_ptr(void) noexcept;

    const CSoAPtr<Ts...>& get_soa_ptr(void) const noexcept;

    void copy(SoAPtr<Ts...> other, const DeviceType other_d_t, const int other_num_items) noexcept;

    template<DeviceType d_t>
    void copy(CSoAData<TypeList<Ts...>, d_t>&& other) noexcept;

    int get_num_items(void) noexcept;

    void resize(const int resize_items) noexcept;

    DeviceType get_device_type(void) noexcept;
};


template<typename... Ts>
void print(SoAData<TypeList<Ts...>, DeviceType::HOST>& data);

template<typename... Ts>
void print(SoAData<TypeList<Ts...>, DeviceType::DEVICE>& data);

template<typename... Ts>
void print(CSoAData<TypeList<Ts...>, DeviceType::HOST>& data);

template<typename... Ts>
void print(CSoAData<TypeList<Ts...>, DeviceType::DEVICE>& data);

}

#include "impl/SoAData.i.cuh"
#endif
