/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
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
 */
#pragma once

namespace gpu {

template<typename T>
void allocate(T*& pointer, size_t num_items);

template<typename T>
void free(T* pointer);

template<typename T>
void copyDeviceToDevice(const T* source, size_t num_items, T* destination);

template<typename T>
void copyHostToDevice(const T* source, size_t num_items, T* destination);

template<typename T>
void copyHostToDevice(T value, T* destination);

template<typename T>
void copyDeviceToHost(const T* source, size_t num_items, T* destination);

template<typename T>
void copyDeviceToHost(const T* source, T& value);

template<typename T>
void memsetZero(const T* pointer, size_t num_items);

template<typename T>
void memsetOne(const T* pointer, size_t num_items);

template<typename T>
T reduce(const T* input, size_t num_items);

template<typename T>
void excl_prefixsum(const T* input, size_t num_items, T* output);

} // namespace gpu

//==============================================================================

namespace host {

template<typename T>
void allocate(T*& pointer, size_t num_items);

template<typename T>
void free(T*& pointer);

template<typename T>
void copyHostToHost(const T* input, size_t num_items, T* output);

template<typename T>
void memsetZero(T* pointer, size_t num_items);

template<typename T>
void memsetOne(T* pointer, size_t num_items);

template<typename T>
T reduce(const T* input, size_t num_items);

template<typename T>
void excl_prefixsum(const T* input, size_t num_items, T* output);

} // namespace host

#include "Core/StandardAPI.i.hpp"
