/**
 * @internal
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
 *
 * @file
 */
#pragma once

#include <cuda_runtime.h>   //cudaError_t, cudaFree
#include <array>            //std::array
#include <limits>           //std::numeric_limits

#if !defined(NO_CHECK_CUDA_ERROR)
    #define CHECK_CUDA_ERROR                                                   \
        {                                                                      \
            cudaDeviceSynchronize();                                           \
            xlib::__getLastCudaError(__FILE__, __LINE__, __func__);            \
        }
    /*#define CUDA_ERROR(msg)                                                  \
        {                                                                      \
            cudaDeviceSynchronize();                                           \
            xlib::__getLastCudaError(msg, __FILE__, __LINE__, __func__);       \
        }*/
#else
    #define CHECK_CUDA_ERROR
    //#define CUDA_ERROR(msg)
#endif

#define SAFE_CALL(function)                                                    \
    {                                                                          \
        xlib::__safe_call(function, __FILE__, __LINE__, __func__);             \
    }

#define CHECK_CUDA_ERROR2                                                      \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        xlib::__getLastCudaError(__FILE__, __LINE__, __func__);                \
    }

namespace xlib {

enum THREAD_GROUP { VOID = 0, WARP, BLOCK };

template<typename T>
struct numeric_limits {         // available in CUDA kernels
    static const T    min = std::numeric_limits<T>::min();
    static const T    max = std::numeric_limits<T>::max();
    static const T lowest = std::numeric_limits<T>::lowest();
};

//------------------------------------------------------------------------------

void __getLastCudaError(const char* file, int line, const char* func_name);

void __safe_call(cudaError_t error, const char* file, int line,
                 const char* func_name);

void __cudaErrorHandler(cudaError_t error, const char* error_message,
                        const char* file, int line, const char* func_name);

class DeviceProperty {
    public:
        static int num_SM();
    private:
        static int NUM_OF_STREAMING_MULTIPROCESSOR;
};

template<int SIZE>
struct CuFreeAtExit {
    template<typename... TArgs>
    explicit CuFreeAtExit(TArgs... args) noexcept;

    ~CuFreeAtExit() noexcept;
private:
    const std::array<void*, SIZE> _tmp;
};

void device_info(int device_id = 0);

} // namespace xlib

#include "impl/CudaUtil.i.cuh"
