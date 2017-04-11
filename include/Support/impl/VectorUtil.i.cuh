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

template<> struct make2_str<char>      {
    using type = char2;
    __host__ __device__ __forceinline__
    static type get(char a, char b) {
        return make_char2(a, b);
    }
};
template<> struct make2_str<short>     {
    using type = short2;
    __host__ __device__ __forceinline__
    static type get(short a, short b) {
        return make_short2(a, b);
    }
};
template<> struct make2_str<int>       {
    using type = int2;
    __host__ __device__ __forceinline__
    static type get(int a, int b) {
        return make_int2(a, b);
    }
};
template<> struct make2_str<unsigned>  {
    using type = uint2;
    __host__ __device__ __forceinline__
    static type get(unsigned a, unsigned b) {
        return make_uint2(a, b);
    }
};
template<> struct make2_str<float>     {
    using type = float2;
    __host__ __device__ __forceinline__
    static type get(float a, float b) {
        return make_float2(a, b);
    }
};
template<> struct make2_str<double>    {
    using type = double2;
    __host__ __device__ __forceinline__
    static type get(double a, double b) {
        return make_double2(a, b);
    }
};
template<> struct make2_str<long long int> {
    using type = long2;
    __host__ __device__ __forceinline__
    static type get(long long int a, long long int b) {
        return make_long2(a, b);
    }
};
template<> struct make2_str<int64_t> {
    using type = long2;
    __host__ __device__ __forceinline__
    static type get(long long int a, long long int b) {
        return make_long2(a, b);
    }
};
template<> struct make2_str<long long unsigned> {
    using type = ulong2;
    __host__ __device__ __forceinline__
    static type get(long long unsigned a, long long unsigned b) {
        return make_ulong2(a, b);
    }
};
template<> struct make2_str<uint64_t> {
    using type = ulong2;
    __host__ __device__ __forceinline__
    static type get(long long unsigned a, long long unsigned b) {
        return make_ulong2(a, b);
    }
};



template<typename T>
__host__ __device__ __forceinline__
typename make2_str<T>::type make2(T a, T b) {
    return make2_str<T>::get(a, b);
}

} // namespace xlib
