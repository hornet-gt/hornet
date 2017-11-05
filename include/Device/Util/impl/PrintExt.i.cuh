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
#include "Host/PrintExt.hpp"             //xlib::printArray
#include "Device/Util/SafeCudaAPI.cuh"   //cuMemcpyFromSymbol

namespace xlib {
namespace gpu {

template<typename T>
void printArray(const T* d_array, size_t size, const std::string& title,
                const std::string& sep) noexcept {
    auto h_array = new T[size];
    cuMemcpyToHost(d_array, size, h_array);
    xlib::printArray(h_array, size, title, sep);
    delete[] h_array;
}

template<typename T, int SIZE>
void printArray(const T (&d_array)[SIZE], const std::string& title,
                const std::string& sep) noexcept {
    auto h_array = new T[SIZE];
    cuMemcpyFromSymbol(d_array, h_array);

    xlib::printArray(h_array, SIZE, title, sep);
    delete[] h_array;
}

template<typename T>
void printSymbol(const T& d_symbol, const std::string& title) noexcept {
    T h_data;
    cuMemcpyFromSymbol(d_symbol, h_data);

    std::cout << title << h_data << std::endl;
}

//==============================================================================

template<typename T>
void printMatrix(const T* d_matrix, size_t rows, size_t cols,
                 const std::string& title) noexcept {
    auto h_matrix = new T[rows * cols];
    cuMemcpyToHost(d_matrix, rows * cols, h_matrix);

    xlib::printMatrix(h_matrix, rows, cols, cols, title);
    delete[] h_matrix;
}

template<typename T>
void printMatrix(const T* d_matrix, size_t rows, size_t cols, size_t ld_cols,
                 const std::string& title) noexcept {
    auto h_matrix = new T[rows * ld_cols];
    cuMemcpyToHost(d_matrix, rows * ld_cols, h_matrix);

    xlib::printMatrix(h_matrix, rows, cols, ld_cols, title);
    delete[] h_matrix;
}

template<typename T>
void printMatrixCM(const T* d_matrix, size_t rows, size_t cols,
                   const std::string& title) noexcept {
    auto h_matrix = new T[rows * cols];
    cuMemcpyToHost(d_matrix, rows * cols, h_matrix);

    xlib::printMatrixCM(h_matrix, rows, cols, rows, title);
    delete[] h_matrix;
}

template<typename T>
void printMatrixCM(const T* d_matrix, size_t rows, size_t cols, size_t ld_rows,
                   const std::string& title) noexcept {
    auto h_matrix = new T[ld_rows * cols];
    cuMemcpyToHost(d_matrix, ld_rows * cols, h_matrix);

    xlib::printMatrixCM(h_matrix, rows, cols, ld_rows, title);
    delete[] h_matrix;
}

//------------------------------------------------------------------------------

template<typename T>
void printMatrixRowsCM(const T* d_matrix, size_t rows, size_t cols,
                       size_t first_row, size_t last_row,
                       const std::string& title) noexcept {
    auto h_matrix = new T[rows * cols];
    cuMemcpyToHost(d_matrix, rows * cols, h_matrix);

    last_row = (last_row == 0) ? rows : last_row;
    xlib::printMatrixCM(h_matrix + first_row, last_row - first_row, cols,
                        rows, title);
    delete[] h_matrix;
}

template<typename T>
void printMatrixRowsCM(const T* d_matrix, size_t rows, size_t cols,
                       size_t first_row, const std::string& title) noexcept {
    printMatrixRowsCM(d_matrix, rows, cols, first_row, 0, title);
}

template<typename T>
void printMatrixColumnsCM(const T* d_matrix, size_t rows, size_t cols,
                          size_t first_col, size_t last_col,
                          const std::string& title) noexcept {
    auto h_matrix = new T[rows * cols];
    cuMemcpyToHost(d_matrix, rows * cols, h_matrix);

    xlib::printMatrixCM(h_matrix + first_col * rows, rows,
                        last_col - first_col, rows, title);
    delete[] h_matrix;
}

//------------------------------------------------------------------------------

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, const char* string) noexcept {
    printf("%s", string);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, uint64_t value) noexcept {
    printf("%llu", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, int64_t value) noexcept {
    printf("%lld", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, long long int value) noexcept {
    printf("%lld", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, long long unsigned value)
                             noexcept {
    printf("%llu", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, int value) noexcept {
    printf("%d", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, unsigned value) noexcept {
    printf("%u", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, short value) noexcept {
    printf("%hd", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, unsigned short value) noexcept {
    printf("%hu", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, char value) noexcept {
    printf("%c", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, unsigned char value) noexcept {
    printf("%hhu", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, float value) noexcept {
    printf("%f", value);
    return obj;
}

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, double value) noexcept {
    printf("%f", value);
    return obj;
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<std::is_pointer<T>::value, const Cout&>::type
operator<<(const Cout& obj, const T pointer) noexcept {
    printf("0x%llX", pointer);
    return obj;
}

} // namespace gpu
} // namespace xlib
