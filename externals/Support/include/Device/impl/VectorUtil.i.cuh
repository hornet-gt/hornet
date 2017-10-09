/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
#include <limits>

namespace std {

template<>
class numeric_limits<int2> {
public:
    static int2 max() {
        return make_int2(std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());
    }
};

template<>
class numeric_limits<int3> {
public:
    static int3 max() {
        return make_int3(std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());
    }
};

template<>
class numeric_limits<int4> {
public:
    static int4 max() {
        return make_int4(std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());
    }
};

} // namespace std

//------------------------------------------------------------------------------

inline std::ostream& operator<< (std::ostream& out, const int2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const uint2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<<(std::ostream& out, const int4& value) {
    out << "(" << value.x << "," << value.y << ","
        << value.z << "," << value.w << ")";
    return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulong2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator << (std::ostream& out, const long2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const int2& A, const int2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const int2& A, const int2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const int2& A, const int2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const int2& A, const int2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const int2& A, const int2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const int2& A, const int2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const uint2& A, const uint2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const uint2& A, const uint2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const uint2& A, const uint2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const uint2& A, const uint2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const uint2& A, const uint2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const uint2& A, const uint2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const ulong2& A, const ulong2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const ulong2& A, const ulong2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const ulong2& A, const ulong2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const ulong2& A, const ulong2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const ulong2& A, const ulong2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const ulong2& A, const ulong2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const long2& A, const long2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const long2& A, const long2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const long2& A, const long2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const long2& A, const long2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const long2& A, const long2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const long2& A, const long2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const int4& A, const int4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const int4& A, const int4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const int4& A, const int4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const int4& A, const int4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const int4& A, const int4& B) {
    return A.x >= B.x && A.y >= B.y & A.z >= B.z & A.w >= B.w;
}

HOST_DEVICE bool operator> (const int4& A, const int4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

// =============================================================================

namespace xlib {

template<>
struct Make2Str<char> {
    using type = char2;

    __host__ __device__ __forceinline__
    static type get(char a, char b) {
        return make_char2(a, b);
    }
};

template<>
struct Make2Str<unsigned char> {
    using type = uchar2;

    __host__ __device__ __forceinline__
    static type get(char a, char b) {
        return make_uchar2(a, b);
    }
};

template<>
struct Make2Str<short> {
    using type = short2;

    __host__ __device__ __forceinline__
    static type get(short a, short b) {
        return make_short2(a, b);
    }
};

template<>
struct Make2Str<unsigned short> {
    using type = ushort2;

    __host__ __device__ __forceinline__
    static type get(short a, short b) {
        return make_ushort2(a, b);
    }
};

template<>
struct Make2Str<int> {
    using type = int2;

    __host__ __device__ __forceinline__
    static type get(int a, int b) {
        return make_int2(a, b);
    }
};

template<>
struct Make2Str<unsigned> {
    using type = uint2;
    __host__ __device__ __forceinline__
    static type get(unsigned a, unsigned b) {
        return make_uint2(a, b);
    }
};

template<>
struct Make2Str<long long int> {
    using type = long2;

    __host__ __device__ __forceinline__
    static type get(long long int a, long long int b) {
        return make_long2(a, b);
    }
};

template<>
struct Make2Str<int64_t> {
    using type = long2;

    __host__ __device__ __forceinline__
    static type get(long long int a, long long int b) {
        return make_long2(a, b);
    }
};

template<>
struct Make2Str<long long unsigned> {
    using type = ulong2;

    __host__ __device__ __forceinline__
    static type get(long long unsigned a, long long unsigned b) {
        return make_ulong2(a, b);
    }
};

template<>
struct Make2Str<uint64_t> {
    using type = ulong2;

    __host__ __device__ __forceinline__
    static type get(long long unsigned a, long long unsigned b) {
        return make_ulong2(a, b);
    }
};

template<>
struct Make2Str<float> {
    using type = float2;
    __host__ __device__ __forceinline__
    static type get(float a, float b) {
        return make_float2(a, b);
    }
};

template<> struct Make2Str<double> {
    using type = double2;
    __host__ __device__ __forceinline__
    static type get(double a, double b) {
        return make_double2(a, b);
    }
};



template<typename T>
__host__ __device__ __forceinline__
typename Make2Str<T>::type make2(T a, T b) {
    return Make2Str<T>::get(a, b);
}

} // namespace xlib
