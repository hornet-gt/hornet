/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
 *   - **fast debbuging**:
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

#include "Device/CudaUtil.cuh"  //__cudaErrorHandler
#include "Host/Basic.hpp"       //xlib::byte_t
#include "Host/Numeric.hpp"     //xlib::upper_approx
#include <cassert>                      //std::assert
#include <utility>                      //std::forward

#if defined(NEVER_DEFINED)
    #include "SafeFunctions_.cuh"
#endif

///@cond
#define cuMalloc(...)                                                          \
    xlib::detail::cuMallocAux(__FILE__, __LINE__, __func__, __VA_ARGS__)        \

#define cuMallocHost(...)                                                      \
    xlib::detail::cuMallocHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__)    \

#define cuFree(...)                                                            \
    xlib::detail::cuFreeAux(__FILE__, __LINE__, __func__, __VA_ARGS__)          \
//------------------------------------------------------------------------------

#define cuMemcpyDeviceToDevice(...)                                            \
    xlib::detail::cuMemcpyDeviceToDeviceAux(__FILE__, __LINE__,__func__,        \
                                            __VA_ARGS__)                       \

#define cuMemcpyToDevice(...)                                                  \
    xlib::detail::cuMemcpyToDeviceAux(__FILE__, __LINE__,__func__, __VA_ARGS__) \

#define cuMemcpyToDeviceAsync(...)                                             \
    xlib::detail::cuMemcpyToDeviceAsyncAux(__FILE__,  __LINE__, __func__,       \
                                           __VA_ARGS__)                        \

#define cuMemcpyToHost(...)                                                    \
    xlib::detail::cuMemcpyToHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__)  \

#define cuMemcpyToHostAsync(...)                                               \
    xlib::detail::cuMemcpyToHostAsyncAux(__FILE__, __LINE__, __func__,          \
                                         __VA_ARGS__)                          \
//------------------------------------------------------------------------------

#define cuMemcpyToSymbol(...)                                                  \
    xlib::detail::cuMemcpyToSymbolAux(__FILE__, __LINE__,__func__, __VA_ARGS__) \

#define cuMemcpyToSymbolAsync(...)                                             \
    xlib::detail::cuMemcpyToSymbolAsyncAux(__FILE__, __LINE__,__func__,         \
                                           __VA_ARGS__)                        \

#define cuMemcpyFromSymbol(...)                                                \
    xlib::detail::cuMemcpyFromSymbolAux(__FILE__, __LINE__,__func__,            \
                                        __VA_ARGS__)                           \

#define cuMemcpyFromSymbolAsync(...)                                           \
    xlib::detail::cuMemcpyFromSymbolAsyncAux(__FILE__, __LINE__,__func__,       \
                                             __VA_ARGS__)                      \
//------------------------------------------------------------------------------

#define cuMemset0x00(...)                                                      \
    xlib::detail::cuMemset0x00Aux(__FILE__, __LINE__, __func__, __VA_ARGS__)    \

#define cuMemset0xFF(...)                                                      \
    xlib::detail::cuMemset0xFFAux(__FILE__, __LINE__, __func__, __VA_ARGS__)    \

//==============================================================================
//==============================================================================

namespace xlib {
namespace detail {

template<typename T>
void cuMallocAux(const char* file, int line, const char* func_name,
                 T*& ptr, size_t num_items) {
    assert(num_items > 0);
    xlib::__cudaErrorHandler(cudaMalloc(&ptr, num_items * sizeof(T)),
                             "cudaMalloc", file, line, func_name);
}

//------------------------------------------------------------------------------
template<typename T>
size_t byte_size(T* ptr, size_t num_items) {
    return num_items * sizeof(T);
}

template<typename T, typename... TArgs>
size_t byte_size(T* ptr, size_t num_items, TArgs... args) {
    return xlib::upper_approx<512>(num_items * sizeof(T)) + byte_size(args...);
}

template<typename T>
void set_ptr(xlib::byte_t* base_ptr, T*& ptr, size_t) {
    ptr = reinterpret_cast<T*>(base_ptr);
}

template<typename T, typename... TArgs>
void set_ptr(xlib::byte_t* base_ptr, T*& ptr, size_t num_items, TArgs... args) {
    ptr = reinterpret_cast<T*>(base_ptr);
    set_ptr(base_ptr + xlib::upper_approx<512>(num_items * sizeof(T)), args...);
}

template<typename... TArgs>
void cuMallocAux(const char* file, int line, const char* func_name,
                 TArgs&&... args) {
    size_t num_bytes = byte_size(args...);
    assert(num_bytes > 0);
    xlib::byte_t* base_ptr;
    xlib::__cudaErrorHandler(cudaMalloc(&base_ptr, num_bytes), "cudaMalloc",
                             file, line, func_name);
    set_ptr(base_ptr, std::forward<TArgs>(args)...);
}

template<typename... TArgs>
void cuMallocHostAux(const char* file, int line, const char* func_name,
                     TArgs&&... args) {
    size_t num_bytes = byte_size(args...);
    assert(num_bytes > 0);
    xlib::byte_t* base_ptr;
    xlib::__cudaErrorHandler(cudaMallocHost(&base_ptr, num_bytes), "cudaMalloc",
                             file, line, func_name);
    set_ptr(base_ptr, std::forward<TArgs>(args)...);
}

//------------------------------------------------------------------------------
template<typename T>
void cuFreeAux(const char* file, int line, const char* func_name, T* ptr) {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R*>(ptr);
    xlib::__cudaErrorHandler(cudaFree(ptr1), "cudaFree", file, line, func_name);
}

template<typename T, typename... TArgs>
void cuFreeAux(const char* file, int line, const char* func_name,
               T* ptr, TArgs*... ptrs) {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R*>(ptr);
    xlib::__cudaErrorHandler(cudaFree(ptr1), "cudaFree", file, line, func_name);
    cuFreeAux(file, line, func_name, ptrs...);
}
/*
template<typename... TArgs>
void cuFreeAux(const char* file, int line, const char* func_name,
                      TArgs*... ptrs) {
    std::array<const void*, sizeof...(ptrs)> array =
                                   {{ reinterpret_cast<const void*>(ptrs)... }};
    for (const auto& it : array) {
        xlib::__cudaErrorHandler(cudaFree(it), "cudaFree",
                                 file, line, func_name);
    }
}*/
//------------------------------------------------------------------------------

template<typename T>
void cuMemset0x00Aux(const char* file, int line, const char* func_name,
                     T* ptr, size_t num_items = 1) {
    assert(num_items > 0 && ptr != nullptr);
    xlib::__cudaErrorHandler(cudaMemset(ptr, 0x00, num_items * sizeof(T)),
                             "cudaMemset(0x00)", file, line, func_name);
}

template<typename T>
void cuMemset0x00Aux(const char* file, int line, const char* func_name,T& ref) {
    xlib::__cudaErrorHandler(cudaMemset(ref, 0x00, sizeof(T)),
                             "cudaMemset(0x00)", file, line, func_name);
}

template<typename T>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                     T* ptr, size_t num_items = 1) {
    assert(num_items > 0 && ptr != nullptr);
    xlib::__cudaErrorHandler(cudaMemset(ptr, 0xFF, num_items * sizeof(T)),
                             "cudaMemset(0xFF)", file, line, func_name);
}

template<typename T>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,T& ref) {
    xlib::__cudaErrorHandler(cudaMemset(ref, 0xFF, sizeof(T)),
                             "cudaMemset(0xFF)", file, line, func_name);
}
//==============================================================================
////////////////////////
//  cuMemcpyToDevice  //
////////////////////////

//Pointer to Pointer
template<typename T>
void cuMemcpyToDeviceAux(const char* file, int line, const char* func_name,
                         const T* input, size_t num_items, T* output) {
    assert(input != nullptr && output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, input, num_items * sizeof(T),
                                        cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}

//Reference to Pointer
template<typename T>
void cuMemcpyToDeviceAux(const char* file, int line, const char* func_name,
                         const T& input, T* output) {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, &input, sizeof(T),
                                        cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}

//Fixed Array to Pointer
template<typename T, int SIZE>
void cuMemcpyToDeviceAux(const char* file, int line,
                                const char* func_name,
                                const T (&input)[SIZE], T* output) {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, &input, SIZE * sizeof(T),
                                        cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}

//==============================================================================
//////////////////////////////
//  cuMemcpyDeviceToDevice  //
//////////////////////////////

//Pointer to Pointer
template<typename T>
void cuMemcpyDeviceToDeviceAux(const char* file, int line,
                               const char* func_name, const T* input,
                               size_t num_items, T* output) {
    assert(input != nullptr && output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, input, num_items * sizeof(T),
                                        cudaMemcpyDeviceToDevice),
                            "cudaMemcpyDeviceToDevice)", file, line, func_name);
}
/*
//Reference to Pointer
template<typename T>
void cuMemcpyDeviceToDeviceAux(const char* file, int line, const char* func_name,
                         const T& input, T* output) {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, &input, sizeof(T),
                                        cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}

//Fixed Array to Pointer
template<typename T, int SIZE>
void cuMemcpyDeviceToDeviceAux(const char* file, int line,
                                const char* func_name,
                                const T (&input)[SIZE], T* output) {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, &input, SIZE * sizeof(T),
                                        cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}*/

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpyToDeviceAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t num_items, T* output) {
    assert(num_items > 0 && input != nullptr && output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyAsync(output, input,
                                             num_items * sizeof(T),
                                             cudaMemcpyHostToDevice),
                            "cudaMemcpyAsync(ToDevice)", file, line, func_name);
}

//Reference to Pointer
template<typename T>
void cuMemcpyToDeviceAsyncAux(const char* file, int line, const char* func_name,
                              const T& input, T* output) {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyAsync(output, &input, sizeof(T),
                                             cudaMemcpyHostToDevice),
                            "cudaMemcpyAsync(ToDevice)", file, line, func_name);
}

//Fixed Array to Pointer
template<typename T, int SIZE>
void cuMemcpyToDeviceAsyncAux(const char* file, int line, const char* func_name,
                              const T (&input)[SIZE], T* output) {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyAsync(output, &input, SIZE * sizeof(T),
                                             cudaMemcpyHostToDevice),
                            "cudaMemcpyAsync(ToDevice)", file, line, func_name);
}

//==============================================================================
//////////////////////
//  cuMemcpyToHost  //
//////////////////////

//Pointer To Pointer
template<typename T>
void cuMemcpyToHostAux(const char* file, int line, const char* func_name,
                       const T* input, size_t num_items, T* output) {
    assert(input != nullptr && output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, input, num_items * sizeof(T),
                                        cudaMemcpyDeviceToHost),
                            "cudaMemcpy(ToHost)", file, line, func_name);
}

template<typename T>
void cuMemcpyToHostAux(const char* file, int line, const char* func_name,
                       const T* input, T& output) {
    assert(input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(&output, input, sizeof(T),
                                        cudaMemcpyDeviceToHost),
                            "cudaMemcpy(ToHost)", file, line, func_name);
}

//------------------------------------------------------------------------------

//Pointer To Pointer
template<typename T>
void cuMemcpyToHostAsyncAux(const char* file, int line, const char* func_name,
                            const T* input, size_t num_items, T* output) {
    assert(input != nullptr && output!= nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyAsync(output, input,
                                             num_items * sizeof(T),
                                             cudaMemcpyDeviceToHost),
                            "cudaMemcpyAsync(ToHost)", file, line, func_name);
}

template<typename T>
void cuMemcpyToHostAsyncAux(const char* file, int line, const char* func_name,
                            const T* input, T& output) {
    assert(input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyAsync(&output, input, sizeof(T),
                                             cudaMemcpyDeviceToHost),
                            "cudaMemcpyAsync(ToHost)", file, line, func_name);
}

//==============================================================================
////////////////////////
//  cuMemcpyToSymbol  //
////////////////////////

//Reference To Reference
template<typename T>
void cuMemcpyToSymbolAux(const char* file, int line, const char* func_name,
                         const T& input, T& symbol) {
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, &input, sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

//Pointer To Fixed Array
template<typename T, int SIZE>
void cuMemcpyToSymbolAux(const char* file, int line, const char* func_name,
                         const T* input, size_t num_items, T (&symbol)[SIZE],
                         size_t item_offset = 0) {

    assert(num_items + item_offset <= SIZE &&  input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, input,
                                                num_items * sizeof(T),
                                                item_offset * sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

//------------------------------------------------------------------------------

//Reference To Reference
template<typename T>
void cuMemcpyToSymbolAsyncAux(const char* file, int line, const char* func_name,
                              const T& input, T& symbol) {

    xlib::__cudaErrorHandler(cudaMemcpyToSymbolAsync(symbol, &input, sizeof(T)),
                             "cudaMemcpyToSymbolAsync", file, line, func_name);
}

//Pointer To Fixed Array
template<typename T, int SIZE>
void cuMemcpyToSymbolAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t num_items,
                              T (&symbol)[SIZE], size_t item_offset = 0) {

    assert(num_items + item_offset <= SIZE && input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyToSymbolAsync(symbol, input,
                                                     num_items * sizeof(T),
                                                     item_offset * sizeof(T)),
                             "cudaMemcpyToSymbolAsync", file, line, func_name);
}

//==============================================================================
////////////////////////
// cuMemcpyFromSymbol //
////////////////////////

//Reference To Reference
template<typename T>
void cuMemcpyFromSymbolAux(const char* file, int line, const char* func_name,
                           const T& symbol, T& output) {

    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(&output, symbol, sizeof(T)),
                             "cudaMemcpyFromSymbol", file, line, func_name);
}

//------------------------------------------------------------------------------

//Reference To Reference
template<typename T>
void cuMemcpyFromSymbolAsyncAux(const char* file, int line,
                                const char* func_name, const T& symbol,
                                T& output) {

    xlib::__cudaErrorHandler(cudaMemcpyFromSymbolAsync(&output, symbol,
                                                       sizeof(T)),
                           "cuMemcpyFromSymbolAsyncAux", file, line, func_name);
}

///@endcond

} // namespace detail
} // namespace xlib
