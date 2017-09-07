/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
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

#include <iostream>

namespace hornet_alg {

template<typename T>
class HostDeviceVar {
public:
    explicit HostDeviceVar() noexcept {
        cuMalloc(_d_value_ptr, 1);
    }

    explicit HostDeviceVar(const T& value) noexcept : _value(value) {
        cuMalloc(_d_value_ptr, 1);
    }

    HostDeviceVar(const HostDeviceVar& obj) noexcept :
                                            _value(obj._value),
                                            _d_value_ptr(obj._d_value_ptr),
                                            _is_kernel(true) {
        cuMemcpyToDeviceAsync(_value, _d_value_ptr);
        obj._first_eval = false;
    }

    ~HostDeviceVar() noexcept {
        if (_is_kernel)
            cuFree(_d_value_ptr);
    }

    __host__ __device__ __forceinline__
    operator T() const noexcept {
#if !defined(__CUDA_ARCH__)
        cuMemcpyToHostAsync(_d_value_ptr, _value);
#endif
        return _value;
    }

    __host__ __device__ __forceinline__
    operator T() noexcept {
#if !defined(__CUDA_ARCH__)
        if (!_first_eval)
            cuMemcpyToHostAsync(_d_value_ptr, _value);
        _first_eval = false;
#endif
        return _value;
    }

    __device__ __forceinline__
    T& ref() noexcept {
        return &*_d_value_ptr;
    }

    __device__ __forceinline__
    T* ptr() noexcept {
        return _d_value_ptr;
    }

    __host__ __device__ __forceinline__
    const T& operator=(const T& value) noexcept {
#if defined(__CUDA_ARCH__)
        *_d_value_ptr = value;
#else
        cuMemcpyToDeviceAsync(_value, _d_value_ptr);
#endif
        return value;
    }

    template<typename R>
    friend inline
    typename std::enable_if<xlib::is_stream_insertable<R>::value,
                            std::ostream&>::type
    operator<<(std::ostream& os, const HostDeviceVar<R>& obj) {
        os << obj;
        return os;
    }

private:
    T            _value;
    T*           _d_value_ptr { nullptr };
    bool         _is_kernel   { false };
    mutable bool _first_eval  { true };
};

} // namespace hornet_alg
