/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
 *
 * @brief Improved CUDA APIs
 * @details Advatages:                                                      <br>
 *   - **clear semantic**: input, then output (google style)
 *   - **type checking**:
 *      - input and output must have the same type T
 *      - const checking for inputs
 *      - device symbols must be references
 *   - **no byte object sizes**: the number of bytes is  determined by looking
 *       the parameter type T
 *   - **fast debugging**:
 *      - in case of error the macro provides the file name, the line, the
 *        name of the function where it is called, and the API name that fail
 *      - assertion to check null pointers and num_items == 0
 *      - assertion to check every CUDA API errors
 *      - additional info: cudaMalloc fail -> what is the available memory?
 *   - **direct argument passing** of constant values. E.g.                 <br>
 *       \code{.cu}
 *        cuMemcpyToSymbol(false, d_symbol); //d_symbol must be bool
 *       \endcode
 *   - much **less verbose**
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
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
#pragma once

#include "Device/Util/CudaUtil.cuh" //__cudaErrorHandler
#include "Host/Basic.hpp"           //xlib::byte_t
#include "Host/Numeric.hpp"         //xlib::upper_approx
#include <cassert>                  //assert
#include <utility>                  //std::forward

#if defined(NEVER_DEFINED)
    #include "SafeFunctions_.cuh"
#endif

///@cond

#define cuGetSymbolAddress(...)                                                \
    xlib::detail::cucuGetSymbolAddressAux(__FILE__, __LINE__, __func__,        \
                                          __VA_ARGS__)                         \

#define cuMalloc(...)                                                          \
    xlib::detail::cuMallocAux(__FILE__, __LINE__, __func__, __VA_ARGS__)       \

#define cuMalloc2D(...)                                                        \
    xlib::detail::cuMalloc2DAux(__FILE__, __LINE__, __func__, __VA_ARGS__)     \

#define cuMallocHost(...)                                                      \
    xlib::detail::cuMallocHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__)   \

#define cuFree(...)                                                            \
    xlib::detail::cuFreeAux(__FILE__, __LINE__, __func__, __VA_ARGS__)         \

#define cuFreeHost(...)                                                        \
    xlib::detail::cuFreeHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__)     \

//------------------------------------------------------------------------------

#define cuMemcpyDevToDev(...)                                                  \
    xlib::detail::cuMemcpyDevToDevAux(__FILE__, __LINE__,__func__,             \
                                            __VA_ARGS__)                       \

#define cuMemcpyToDevice(...)                                                  \
    xlib::detail::cuMemcpyToDeviceAux(__FILE__, __LINE__,__func__, __VA_ARGS__)\

#define cuMemcpyToHost(...)                                                    \
    xlib::detail::cuMemcpyToHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__) \

//------------------------------------------------------------------------------

#define cuMemcpyToSymbol(...)                                                  \
    xlib::detail::cuMemcpyToSymbolAux(__FILE__, __LINE__,__func__, __VA_ARGS__)\

#define cuMemcpyFromSymbol(...)                                                \
    xlib::detail::cuMemcpyFromSymbolAux(__FILE__, __LINE__,__func__,           \
                                        __VA_ARGS__)                           \

//------------------------------------------------------------------------------

#define cuMemset0x00(...)                                                      \
    xlib::detail::cuMemset0x00Aux(__FILE__, __LINE__, __func__, __VA_ARGS__)   \

#define cuMemset0xFF(...)                                                      \
    xlib::detail::cuMemset0xFFAux(__FILE__, __LINE__, __func__, __VA_ARGS__)   \

#define cuMemset(...)                                                          \
    xlib::detail::cuMemsetAux(__FILE__, __LINE__, __func__, __VA_ARGS__)       \

//------------------------------------------------------------------------------

#define cuMemset2D0x00(...)                                                    \
    xlib::detail::cuMemset2D0x00Aux(__FILE__, __LINE__, __func__, __VA_ARGS__) \

#define cuMemset2D0xFF(...)                                                    \
    xlib::detail::cuMemset2D0xFFAux(__FILE__, __LINE__, __func__, __VA_ARGS__) \

#define cuMemset2D(...)                                                        \
    xlib::detail::cuMemset2DAux(__FILE__, __LINE__, __func__, __VA_ARGS__)     \

//------------------------------------------------------------------------------

#define cuMemcpy2DToDevice(...)                                                \
    xlib::detail::cuMemcpy2DToDeviceAux(__FILE__,  __LINE__, __func__,         \
                                        __VA_ARGS__)                           \

#define cuMemcpy2DToHost(...)                                                  \
    xlib::detail::cuMemcpy2DToHostAux(__FILE__,  __LINE__, __func__,           \
                                      __VA_ARGS__)                             \

#define cuMemcpy2DDevToDev(...)                                                \
    xlib::detail::cuMemcpy2DDevToDevAux(__FILE__,  __LINE__, __func__,         \
                                        __VA_ARGS__)                           \

//==============================================================================
//==============================================================================

namespace xlib {
namespace detail {

template<typename T>
void cucuGetSymbolAddressAux(const char* file, int line, const char* func_name,
                             T& symbol, T*& ptr) noexcept {
    xlib::__cudaErrorHandler(cudaGetSymbolAddress((void**)&ptr, symbol),
                             "cudaGetSymbolAddress", file, line, func_name);
}

template<typename T, int SIZE>
void cucuGetSymbolAddressAux(const char* file, int line, const char* func_name,
                             T (&symbol)[SIZE], T*& ptr) noexcept {
    xlib::__cudaErrorHandler(cudaGetSymbolAddress((void**)&ptr, symbol),
                             "cudaGetSymbolAddress", file, line, func_name);
}

////////////////
//  cuMalloc  //
////////////////

template<typename T>
void cuMallocAux(const char* file, int line, const char* func_name,
                 T*& ptr, size_t num_items) noexcept {
    assert(num_items > 0);
    xlib::__cudaErrorHandler(cudaMalloc(&ptr, num_items * sizeof(T)),
                             "cudaMalloc", file, line, func_name);
}

//------------------------------------------------------------------------------
template<typename T>
size_t byte_size(T* ptr, size_t num_items) noexcept {
    return num_items * sizeof(T);
}

template<typename T, typename... TArgs>
size_t byte_size(T* ptr, size_t num_items, TArgs... args) noexcept {
    return xlib::upper_approx<512>(num_items * sizeof(T)) + byte_size(args...);
}

template<typename T>
void set_ptr(xlib::byte_t* base_ptr, T*& ptr, size_t) noexcept {
    ptr = reinterpret_cast<T*>(base_ptr);
}

template<typename T, typename... TArgs>
void set_ptr(xlib::byte_t* base_ptr, T*& ptr, size_t num_items, TArgs... args)
             noexcept {
    ptr = reinterpret_cast<T*>(base_ptr);
    set_ptr(base_ptr + xlib::upper_approx<512>(num_items * sizeof(T)), args...);
}

template<typename... TArgs>
void cuMallocAux(const char* file, int line, const char* func_name,
                 TArgs&&... args) noexcept {
    size_t num_bytes = byte_size(args...);
    assert(num_bytes > 0);
    xlib::byte_t* base_ptr;
    xlib::__cudaErrorHandler(cudaMalloc(&base_ptr, num_bytes), "cudaMalloc",
                             file, line, func_name);
    set_ptr(base_ptr, std::forward<TArgs>(args)...);
}

template<typename T>
void cuMalloc2DAux(const char* file, int line, const char* func_name,
                   T*& pointer, size_t rows, size_t cols) noexcept {
    assert(rows > 0 && cols > 0);
    xlib::__cudaErrorHandler(cudaMalloc(&pointer, rows * cols * sizeof(T)),
                                        "cudaMalloc2D", file, line, func_name);
}

template<typename T>
void cuMalloc2DAux(const char* file, int line, const char* func_name,
                   T*& pointer, size_t rows, size_t cols, size_t& pitch)
                   noexcept {
    assert(rows > 0 && cols > 0);
    xlib::__cudaErrorHandler(cudaMallocPitch(&pointer, &pitch,
                                             rows * sizeof(T)), cols,
                                             "cudaMalloc2D",
                                             file, line, func_name);
    assert(pitch % sizeof(T) == 0);
    pitch /= sizeof(T);
}

template<typename... TArgs>
void cuMallocHostAux(const char* file, int line, const char* func_name,
                     TArgs&&... args) noexcept {
    size_t num_bytes = byte_size(args...);
    assert(num_bytes > 0);
    xlib::byte_t* base_ptr;
    xlib::__cudaErrorHandler(cudaMallocHost(&base_ptr, num_bytes), "cudaMalloc",
                             file, line, func_name);
    set_ptr(base_ptr, std::forward<TArgs>(args)...);
}

//------------------------------------------------------------------------------
//////////////
//  cuFree  //
//////////////

template<typename T>
void cuFreeAux(const char* file, int line, const char* func_name, T* ptr)
               noexcept {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R*>(ptr);
    xlib::__cudaErrorHandler(cudaFree(ptr1), "cudaFree", file, line, func_name);
}

template<typename T, typename... TArgs>
void cuFreeAux(const char* file, int line, const char* func_name,
               T* ptr, TArgs*... ptrs) noexcept {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R*>(ptr);
    xlib::__cudaErrorHandler(cudaFree(ptr1), "cudaFree", file, line, func_name);
    cuFreeAux(file, line, func_name, ptrs...);
}

template<typename T>
void cuFreeHostAux(const char* file, int line, const char* func_name, T* ptr)
                   noexcept {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R*>(ptr);
    xlib::__cudaErrorHandler(cudaFreeHost(ptr1), "cudaFreeHost", file, line,
                             func_name);
}

template<typename T, typename... TArgs>
void cuFreeHostAux(const char* file, int line, const char* func_name,
                   T* ptr, TArgs*... ptrs) noexcept {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R*>(ptr);
    xlib::__cudaErrorHandler(cudaFreeHost(ptr1), "cudaFreeHost", file, line,
                             func_name);
    cuFreeHostAux(file, line, func_name, ptrs...);
}

//==============================================================================
////////////////
//  cuMemset  //
////////////////

template<typename T>
void cuMemsetGenericAux(const char* file, int line, const char* func_name,
                        T* ptr, size_t num_items, unsigned char mask) noexcept {
    assert(num_items > 0 && ptr != nullptr);
    char api_name[] = "cudaMemset(0x__)";
    char value1 = static_cast<char>(mask / (0xF));
    char value2 = static_cast<char>(mask % (0xF));
    api_name[13] = (value1 <= '9') ? '0' + value1 : 'A' + value1 - 10;
    api_name[14] = (value2 <= '9') ? '0' + value2 : 'A' + value2 - 10;
    xlib::__cudaErrorHandler(cudaMemset(ptr, mask, num_items * sizeof(T)),
                                        api_name, file, line, func_name);
}

template<typename T>
void cuMemset0x00Aux(const char* file, int line, const char* func_name,
                     T* ptr, size_t num_items = 1) noexcept {
    cuMemsetGenericAux(file, line, func_name, ptr, num_items, 0x00);
}

template<typename T>
void cuMemset0x00Aux(const char* file, int line, const char* func_name,
                     T& symbol) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetGenericAux(file, line, func_name, symbol_address, 1, 0x00);
}

template<typename T, int SIZE>
void cuMemset0x00Aux(const char* file, int line, const char* func_name,
                     T (&symbol)[SIZE]) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetGenericAux(file, line, func_name, symbol_address, SIZE, 0x00);
}

template<typename T>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                     T* ptr, size_t num_items = 1) noexcept {
    cuMemsetGenericAux(file, line, func_name, ptr, num_items, 0xFF);
}

template<typename T>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                     T& symbol) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetGenericAux(file, line, func_name, symbol_address, 1, 0xFF);
}

template<typename T, int SIZE>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                     T (&symbol)[SIZE]) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetGenericAux(file, line, func_name, symbol_address, SIZE, 0xFF);
}

template<typename T>
void cuMemsetAux(const char* file, int line, const char* func_name,
                 T* ptr, size_t num_items, unsigned char mask) noexcept {
    cuMemsetGenericAux(file, line, func_name, ptr, num_items, mask);
}

template<typename T>
void cuMemsetAux(const char* file, int line, const char* func_name,
                 T& symbol, unsigned char mask) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetGenericAux(file, line, func_name, symbol_address, 1, mask);
}

template<typename T, int SIZE>
void cuMemsetAux(const char* file, int line, const char* func_name,
                 T (&symbol)[SIZE], unsigned char mask) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetGenericAux(file, line, func_name, symbol_address, SIZE, mask);
}

//==============================================================================
//////////////////
//  cuMemset2D  //
//////////////////

template<typename T>
void cuMemset2DGenericAux(const char* file, int line, const char* func_name,
                          T* ptr, size_t rows, size_t cols, size_t pitch,
                          unsigned char mask) noexcept {
    assert(ptr != nullptr && rows > 0 && cols > 0 && pitch >= cols);
    char api_name[] = "cudaMemset2D(0x__)";
    char value1 = static_cast<char>(mask / (0xF));
    char value2 = static_cast<char>(mask % (0xF));
    api_name[13] = (value1 <= '9') ? '0' + value1 : 'A' + value1 - 10;
    api_name[14] = (value2 <= '9') ? '0' + value2 : 'A' + value2 - 10;
    xlib::__cudaErrorHandler(cudaMemset2D(ptr, pitch * sizeof(T), mask,
                                          cols * sizeof(T), rows),
                                          api_name, file, line, func_name);
}

template<typename T>
void cuMemset2D0x00Aux(const char* file, int line, const char* func_name,
                       T* ptr, size_t rows, size_t cols, size_t pitch = 0)
                       noexcept {
    pitch = (pitch == 0) ? cols : pitch;
    cuMemset2DGenericAux(file, line, func_name, ptr, rows, cols, pitch, 0x00);
}

template<typename T>
void cuMemset2D0xFFAux(const char* file, int line, const char* func_name,
                       T* ptr, size_t rows, size_t cols, size_t pitch = 0)
                       noexcept {
    pitch = (pitch == 0) ? cols : pitch;
    cuMemset2DGenericAux(file, line, func_name, ptr, rows, cols, pitch, 0xFF);
}

template<typename T>
void cuMemset2DAux(const char* file, int line, const char* func_name,
                   T* ptr, size_t rows, size_t cols, unsigned char mask)
                   noexcept {
    cuMemset2DGenericAux(file, line, func_name, ptr, rows, cols, cols, mask);
}

template<typename T>
void cuMemset2DAux(const char* file, int line, const char* func_name,
                   T* ptr, size_t rows, size_t cols, size_t pitch,
                   unsigned char mask) noexcept {
    cuMemset2DGenericAux(file, line, func_name, ptr, rows, cols, pitch, mask);
}

//==============================================================================
////////////////
//  cuMemcpy  //
////////////////

template<typename T>
void cuMemcpyGenericAux(const char* file, int line, const char* func_name,
                        const T* input, size_t num_items, T* output,
                        cudaMemcpyKind cuda_memcpy_kind) noexcept {
    assert(input != nullptr && output != nullptr);
    if (num_items == 0)
        return;
    const char* api_name[] = { "", "cudaMemcpy(ToDevice)",
                               "cudaMemcpy(ToHost)",
                               "cudaMemcpy(DeviceToDevice)", "" };
    const auto& selected = api_name[static_cast<int>(cuda_memcpy_kind)];
    xlib::__cudaErrorHandler(cudaMemcpy(output, input, num_items * sizeof(T),
                                        cuda_memcpy_kind), selected,
                                        file, line, func_name);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpyToDeviceAux(const char* file, int line, const char* func_name,
                         const T* input, size_t num_items, T* output) noexcept {
    cuMemcpyGenericAux(file, line, func_name, input, num_items, output,
                       cudaMemcpyHostToDevice);
}

template<typename T>
void cuMemcpyToDeviceAux(const char* file, int line, const char* func_name,
                         const T& input, T* output) noexcept {
    cuMemcpyGenericAux(file, line, func_name, &input, 1, output,
                       cudaMemcpyHostToDevice);
}

//Fixed Array to Pointer
template<typename T, int SIZE>
void cuMemcpyToDeviceAux(const char* file, int line, const char* func_name,
                         const T (&input)[SIZE], T* output) noexcept {
    cuMemcpyGenericAux(file, line, func_name, &input, SIZE, output,
                       cudaMemcpyHostToDevice);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpyToHostAux(const char* file, int line, const char* func_name,
                       const T* input, size_t num_items, T* output) noexcept {
    cuMemcpyGenericAux(file, line, func_name, input, num_items, output,
                       cudaMemcpyDeviceToHost);
}

template<typename T>
void cuMemcpyToHostAux(const char* file, int line, const char* func_name,
                       const T* input, T& output) noexcept {
    cuMemcpyGenericAux(file, line, func_name, input, 1, &output,
                       cudaMemcpyDeviceToHost);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpyDevToDevAux(const char* file, int line, const char* func_name,
                         const T* input, size_t num_items, T* output) noexcept {
    cuMemcpyGenericAux(file, line, func_name, input, num_items, output,
                       cudaMemcpyDeviceToDevice);
}

//==============================================================================
//////////////////
//  cuMemcpy2D  //
//////////////////

template<typename T>
void cuMemcpy2DGeneric(const char* file, int line, const char* func_name,
                       const T* input, size_t rows, size_t cols,
                       size_t src_pitch, T* output, size_t dst_pitch,
                       cudaMemcpyKind cuda_memcpy_kind) noexcept {
    assert(input != nullptr && output != nullptr && rows > 0 && cols > 0 &&
           src_pitch >= cols && dst_pitch >= cols);
    const char* api_name[] = { "", "cuda2DMemcpy(ToDevice)",
                               "cuda2DMemcpy(ToHost)",
                               "cuda2DMemcpy(DeviceToDevice)", "" };
    const auto& selected = api_name[static_cast<int>(cuda_memcpy_kind)];
    xlib::__cudaErrorHandler(cudaMemcpy2D(output, dst_pitch * sizeof(T), input,
                                          src_pitch * sizeof(T),
                                          cols * sizeof(T), rows,
                                          cuda_memcpy_kind), selected,
                                          file, line, func_name);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpy2DToDeviceAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols,
                           size_t src_pitch, T* output, size_t dst_pitch)
                           noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, src_pitch,
                      output, dst_pitch, cudaMemcpyHostToDevice);
}

template<typename T>
void cuMemcpy2DToDeviceAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols,
                           T* output, size_t dst_pitch) noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, cols,
                      output, dst_pitch, cudaMemcpyHostToDevice);
}

template<typename T>
void cuMemcpy2DToDeviceAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols,
                           size_t src_pitch, T* output) noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, src_pitch,
                      output, cudaMemcpyHostToDevice);
}

template<typename T>
void cuMemcpy2DToDeviceAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols, T* output)
                           noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, cols,
                      output, cols, cudaMemcpyHostToDevice);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpy2DToHostAux(const char* file, int line, const char* func_name,
                         const T* input, size_t rows, size_t cols,
                         size_t src_pitch, T* output, size_t dst_pitch)
                         noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, src_pitch,
                      output, dst_pitch, cudaMemcpyDeviceToHost);
}

template<typename T>
void cuMemcpy2DToHostAux(const char* file, int line, const char* func_name,
                         const T* input, size_t rows, size_t cols,
                         T* output, size_t dst_pitch) noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, cols,
                      output, dst_pitch, cudaMemcpyDeviceToHost);
}

template<typename T>
void cuMemcpy2DToHostAux(const char* file, int line, const char* func_name,
                         const T* input, size_t rows, size_t cols,
                         size_t src_pitch, T* output) noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, src_pitch,
                      output, cols, cudaMemcpyDeviceToHost);
}

template<typename T>
void cuMemcpy2DToHostAux(const char* file, int line, const char* func_name,
                         const T* input, size_t rows, size_t cols, T* output)
                         noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, cols,
                      output, cols, cudaMemcpyDeviceToHost);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpy2DDevToDevAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols,
                           size_t src_pitch, T* output, size_t dst_pitch)
                           noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, src_pitch,
                      output, dst_pitch, cudaMemcpyDeviceToDevice);
}

template<typename T>
void cuMemcpy2DDevToDevAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols,
                           T* output, size_t dst_pitch) noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, cols,
                      output, dst_pitch, cudaMemcpyDeviceToDevice);
}

template<typename T>
void cuMemcpy2DDevToDevAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols,
                           size_t src_pitch, T* output) noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, src_pitch,
                      output, cols, cudaMemcpyDeviceToDevice);
}

template<typename T>
void cuMemcpy2DDevToDevAux(const char* file, int line, const char* func_name,
                           const T* input, size_t rows, size_t cols, T* output)
                           noexcept {
    cuMemcpy2DGeneric(file, line, func_name, input, rows, cols, cols,
                      output, cols, cudaMemcpyDeviceToDevice);
}

//==============================================================================
////////////////////////
//  cuMemcpyToSymbol  //
////////////////////////

//Reference To Reference
template<typename T>
void cuMemcpyToSymbolAux(const char* file, int line, const char* func_name,
                         const T& input, T& symbol) noexcept {
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, &input, sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

template<typename T, int SIZE>
void cuMemcpyToSymbolAux(const char* file, int line, const char* func_name,
                         const T& input, T (&symbol)[SIZE]) noexcept {
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, &input, sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

//Pointer To Fixed Array
template<typename T, int SIZE>
void cuMemcpyToSymbolAux(const char* file, int line, const char* func_name,
                         const T* input, size_t num_items, T (&symbol)[SIZE],
                         size_t item_offset = 0) noexcept {
    assert(num_items + item_offset <= SIZE && input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, input,
                                                num_items * sizeof(T),
                                                item_offset * sizeof(T)),
                                                "cudaMemcpyToSymbol",
                                                file, line, func_name);
}

//==============================================================================
////////////////////////
// cuMemcpyFromSymbol //
////////////////////////

//Reference To Reference
template<typename T>
void cuMemcpyFromSymbolAux(const char* file, int line, const char* func_name,
                           const T& symbol, T& output) noexcept {
    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(&output, symbol, sizeof(T),
                                                  "cudaMemcpyFromSymbol"),
                                                  file, line, func_name);
}

template<typename T, int SIZE1, int SIZE2>
void cuMemcpyFromSymbolAux(const char* file, int line, const char* func_name,
                           const T (&symbol)[SIZE1], T (&output)[SIZE2])
                           noexcept {
    assert(SIZE1 < SIZE2);
    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(&output, symbol,
                                                  SIZE1 * sizeof(T),
                                                  "cudaMemcpyFromSymbol"),
                                                  file, line, func_name);
}


template<typename T, int SIZE1, int SIZE2>
void cuMemcpyFromSymbolAux(const char* file, int line, const char* func_name,
                           const T (&symbol)[SIZE1], T* output) noexcept {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(output, symbol,
                                                  SIZE1 * sizeof(T)),
                                                  "cudaMemcpyFromSymbol",
                                                  file, line, func_name);
}

///@endcond

} // namespace detail
} // namespace xlib
