/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

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
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

#include "Support/HostDevice.hpp"
#include <ostream>

inline std::ostream& operator << (std::ostream& out, const uint2& value);
inline std::ostream& operator << (std::ostream& out, const int4& value);
inline std::ostream& operator << (std::ostream& out, const ulong2& value);
inline std::ostream& operator << (std::ostream& out, const long2& value);
inline std::ostream& operator << (std::ostream& out, const int2& value);

HOST_DEVICE bool operator== (const int2& A, const int2& B);
HOST_DEVICE bool operator!= (const int2& A, const int2& B);
HOST_DEVICE bool operator<  (const int2& A, const int2& B);
HOST_DEVICE bool operator<= (const int2& A, const int2& B);
HOST_DEVICE bool operator>  (const int2& A, const int2& B);
HOST_DEVICE bool operator>= (const int2& A, const int2& B);

HOST_DEVICE bool operator== (const uint2& A, const uint2& B);
HOST_DEVICE bool operator!= (const uint2& A, const uint2& B);
HOST_DEVICE bool operator<  (const uint2& A, const uint2& B);
HOST_DEVICE bool operator<= (const uint2& A, const uint2& B);
HOST_DEVICE bool operator>  (const uint2& A, const uint2& B);
HOST_DEVICE bool operator>= (const uint2& A, const uint2& B);

HOST_DEVICE bool operator== (const ulong2& A, const ulong2& B);
HOST_DEVICE bool operator!= (const ulong2& A, const ulong2& B);
HOST_DEVICE bool operator<  (const ulong2& A, const ulong2& B);
HOST_DEVICE bool operator<= (const ulong2& A, const ulong2& B);
HOST_DEVICE bool operator>  (const ulong2& A, const ulong2& B);
HOST_DEVICE bool operator>= (const ulong2& A, const ulong2& B);

HOST_DEVICE bool operator== (const long2& A, const long2& B);
HOST_DEVICE bool operator!= (const long2& A, const long2& B);
HOST_DEVICE bool operator<  (const long2& A, const long2& B);
HOST_DEVICE bool operator<= (const long2& A, const long2& B);
HOST_DEVICE bool operator>  (const long2& A, const long2& B);
HOST_DEVICE bool operator>= (const long2& A, const long2& B);

HOST_DEVICE bool operator== (const int4& A, const int4& B);
HOST_DEVICE bool operator!= (const int4& A, const int4& B);
HOST_DEVICE bool operator<  (const int4& A, const int4& B);
HOST_DEVICE bool operator<= (const int4& A, const int4& B);
HOST_DEVICE bool operator>  (const int4& A, const int4& B);
HOST_DEVICE bool operator>= (const int4& A, const int4& B);

namespace xlib {

template<typename T>
struct make2_str {
    using type = typename std::nullptr_t;
};

template<typename T>
__host__ __device__ __forceinline__
typename make2_str<T>::type make2(T a, T b);

} // namespace xlib

#include "impl/VectorUtil.i.cuh"
