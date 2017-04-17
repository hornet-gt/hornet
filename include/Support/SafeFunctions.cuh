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

#include "Support/CudaUtil.cuh"    //__cudaErrorHandler
#include <cassert>                 //std::assert

#if defined(NEVER_DEFINED)
    #include "SafeFunctions_.cuh"
#endif

///@cond
#define cuMalloc(...)                                                          \
    xlib::detail::cuMallocAux(__FILE__, __LINE__, __func__, __VA_ARGS__)       \

#define cuFree(...)                                                            \
    xlib::detail::cuFreeAux(__FILE__, __LINE__, __func__, __VA_ARGS__)         \
//------------------------------------------------------------------------------

#define cuMemcpyToDevice(...)                                                  \
    xlib::detail::cuMemcpyToDeviceAux(__FILE__, __LINE__,__func__, __VA_ARGS__)\

#define cuMemcpyToDeviceAsync(...)                                             \
    xlib::detail::cuMemcpyToDeviceAsyncAux(__FILE__,  __LINE__, __func__,      \
                                           __VA_ARGS__)                        \

#define cuMemcpyToHost(...)                                                    \
    xlib::detail::cuMemcpyToHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__) \

//------------------------------------------------------------------------------

#define cuMemcpyToSymbol(...)                                                  \
    xlib::detail::cuMemcpyToSymbolAux(__FILE__, __LINE__,__func__, __VA_ARGS__)\

#define cuMemcpyToSymbolAsync(...)                                             \
    xlib::detail::cuMemcpyToSymbolAsyncAux(__FILE__, __LINE__,__func__,        \
                                           __VA_ARGS__)                        \

#define cuMemcpyFromSymbol(...)                                                \
    xlib::detail::cuMemcpyFromSymbolAux(__FILE__, __LINE__,__func__,           \
                                        __VA_ARGS__)                           \

#define cuMemcpyFromSymbolAsync(...)                                           \
    xlib::detail::cuMemcpyFromSymbolAsyncAux(__FILE__, __LINE__,__func__,      \
                                             __VA_ARGS__)                      \
//------------------------------------------------------------------------------

#define cuMemset0x00(...)                                                      \
    xlib::detail::cuMemset0x00Aux(__FILE__, __LINE__, __func__, __VA_ARGS__)   \

#define cuMemset0xFF(...)                                                      \
    xlib::detail::cuMemset0xFFAux(__FILE__, __LINE__, __func__, __VA_ARGS__)   \

//==============================================================================
//==============================================================================

namespace xlib {
namespace detail {

template<typename T>
inline void cuMallocAux(const char* file, int line, const char* func_name,
                        T*& ptr, size_t num_items) {
    assert(num_items > 0);
    xlib::__cudaErrorHandler(cudaMalloc(&ptr, num_items * sizeof(T)),
                             "cudaMalloc", file, line, func_name);
}

template<typename T>
inline void cuFreeAux(const char* file, int line, const char* func_name,
                      T* ptr) {
    xlib::__cudaErrorHandler(cudaFree(ptr), "cudaFree", file, line, func_name);
}
template<typename... TArgs, typename T>
inline void cuFreeAux(const char* file, int line, const char* func_name,
                      T* ptr, TArgs... args) {
    xlib::__cudaErrorHandler(cudaFree(ptr), "cudaFree", file, line, func_name);
    cuFreeAux(file, line, func_name, args...);
}
//------------------------------------------------------------------------------

template<typename T>
inline void cuMemset0x00Aux(const char* file, int line, const char* func_name,
                            T* ptr, size_t num_items) {
    assert(num_items > 0 && ptr != nullptr);
    xlib::__cudaErrorHandler(cudaMemset(ptr, 0x00, num_items * sizeof(T)),
                             "cudaMemset(0x00)", file, line, func_name);
}
template<typename T>
inline void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                            T* ptr, size_t num_items) {
    assert(num_items > 0 && ptr != nullptr);
    xlib::__cudaErrorHandler(cudaMemset(ptr, 0xFF, num_items * sizeof(T)),
                             "cudaMemset(0xFF)", file, line, func_name);
}
//------------------------------------------------------------------------------

template<typename T>
inline void cuMemcpyToDeviceAux(const char* file, int line,
                                const char* func_name,
                                const T* input, size_t num_items,
                                T* output) {
    assert(num_items > 0 && input != nullptr && output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, input, num_items * sizeof(T),
                                        cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}
/*template<typename T>
inline void cuMemcpyToDeviceAux(const char* file, int line,
                                const char* func_name,
                                const T& input, T* output) {
    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, &input, sizeof(T),
                                        cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}*/

template<typename T>
inline void cuMemcpyToDeviceAsyncAux(const char* file, int line,
                                     const char* func_name,
                                     const T* input, size_t num_items,
                                     T* output) {
    assert(num_items > 0 && input != nullptr && output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyAsync(output, input,
                                             num_items * sizeof(T),
                                             cudaMemcpyHostToDevice),
                            "cudaMemcpy(ToDevice)", file, line, func_name);
}
//------------------------------------------------------------------------------

template<typename T>
inline void cuMemcpyToHostAux(const char* file, int line,
                              const char* func_name,
                              const T* input, size_t num_items, T* output) {
    assert(num_items > 0 && input != nullptr && output!= nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(output, input, num_items * sizeof(T),
                                        cudaMemcpyDeviceToHost),
                            "cudaMemcpy(ToHost)", file, line, func_name);
}
/*template<typename T>
inline void cuMemcpyToHostAux(const char* file, int line,
                              const char* func_name,
                              const T* input, T& output) {
    assert(input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpy(&output, input, sizeof(T),
                                        cudaMemcpyDeviceToHost),
                            "cudaMemcpy(ToHost)", file, line, func_name);
}*/
//------------------------------------------------------------------------------
////////////////////////
//  cuMemcpyToSymbol  //
////////////////////////

template<typename T>
inline void cuMemcpyToSymbolAux(const char* file, int line,
                                const char* func_name,
                                const T& input, T& symbol) {
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, &input, sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

template<typename T, int SIZE>
inline void cuMemcpyToSymbolAux(const char* file, int line,
                                const char* func_name,
                                const T* input, size_t num_items,
                                T (&symbol)[SIZE],
                                size_t item_offset = 0) {

    assert(num_items > 0 && num_items + item_offset <= SIZE &&
           input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, input,
                                                num_items * sizeof(T),
                                                item_offset * sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

template<typename T, int SIZE>
inline void cuMemcpyToSymbolAux(const char* file, int line,
                                const char* func_name,
                                const T* input, T (&symbol)[SIZE]) {

    assert(input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyToSymbol(symbol, input, SIZE *sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

template<typename T>
inline void cuMemcpyToSymbolAsyncAux(const char* file, int line,
                                     const char* func_name,
                                     const T& input, T& symbol) {

    xlib::__cudaErrorHandler(cudaMemcpyToSymbolAsync(symbol, &input, sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}

template<typename T, int SIZE>
inline void cuMemcpyToSymbolAsyncAux(const char* file, int line,
                                     const char* func_name,
                                     const T* input,
                                     size_t num_items,
                                     T (&symbol)[SIZE],
                                     size_t item_offset = 0) {

    assert(num_items + item_offset <= SIZE && input != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyToSymbolAsync(symbol, input,
                                                     num_items * sizeof(T),
                                                     item_offset * sizeof(T)),
                             "cudaMemcpyToSymbol", file, line, func_name);
}
//------------------------------------------------------------------------------
////////////////////////
// cuMemcpyFromSymbol //
////////////////////////

template<typename T>
inline void cuMemcpyFromSymbolAux(const char* file, int line,
                                  const char* func_name,
                                  const T& symbol, T& output) {

    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(&output, symbol, sizeof(T)),
                             "cudaMemcpyFromSymbol", file, line, func_name);
}
template<typename T, int SIZE>
inline void cuMemcpyFromSymbolAux(const char* file, int line,
                                  const char* func_name,
                                  const T(&symbol)[SIZE], T& output,
                                  size_t item_offset) {

    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(&output, symbol, sizeof(T),
                                                  item_offset * sizeof(T)),
                             "cudaMemcpyFromSymbol", file, line, func_name);
}

template<typename T, int SIZE>
inline void cuMemcpyFromSymbolAux(const char* file, int line,
                                  const char* func_name,
                                  const T(&symbol)[SIZE], size_t num_items,
                                  T* output, size_t item_offset = 0) {

    assert(num_items + item_offset <= SIZE && output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(output, symbol,
                                                  num_items * sizeof(T)),
                             "cudaMemcpyFromSymbol", file, line, func_name);
}

template<typename T, int SIZE>
inline void cuMemcpyFromSymbolAux(const char* file, int line,
                                  const char* func_name,
                                  const T(&symbol)[SIZE], T* output) {

    assert(output != nullptr);
    xlib::__cudaErrorHandler(cudaMemcpyFromSymbol(output, symbol,
                                                  SIZE * sizeof(T)),
                             "cudaMemcpyFromSymbol", file, line, func_name);
}
///@endcond

} // namespace detail
} // namespace xlib
