/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v2
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
#include "Device/Util/SafeCudaAPI.cuh"
#include "Host/Basic.hpp"       //xlib::MB
#include "Host/PrintExt.hpp"    //Color
#include <cuda_runtime_api.h>
#include <iomanip>

int cuGetDeviceCount() noexcept {
    static int device_count = 0;
    if (device_count == 0)
        SAFE_CALL( cudaGetDeviceCount(&device_count) )
    return device_count;
}

void cuSetDevice(int device_index) noexcept {
    SAFE_CALL( cudaSetDevice (device_index) )
}

int cuGetDevice() noexcept {
    int device_index;
    SAFE_CALL( cudaGetDevice (&device_index) )
    return device_index;
}

namespace xlib {
namespace detail {

void getLastCudaError(const char* file, int line, const char* func_name) {
    cudaErrorHandler(cudaGetLastError(), "", file, line, func_name);
}

void safe_call(cudaError_t error, const char* file, int line,
                 const char* func_name) {
    cudaErrorHandler(error, "", file, line, func_name);
}

void cudaErrorHandler(cudaError_t error, const char* error_message,
                      const char* file, int line,
                      const char* func_name) {
    if (cudaSuccess != error) {
        std::cerr << Color::FG_RED << "\nCUDA error\n" << Color::FG_DEFAULT
                  << Emph::SET_UNDERLINE << file
                  << Emph::SET_RESET  << "(" << line << ")"
                  << " [ "
                  << Color::FG_L_CYAN << func_name << Color::FG_DEFAULT
                  << " ] : " << error_message
                  << " -> " << cudaGetErrorString(error)
                  << "(" << static_cast<int>(error) << ")\n";
        if (error == cudaErrorMemoryAllocation) {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            std::cerr << "\nActual allocated memory: " << std::setprecision(1)
                      << std::fixed << (total - free) / xlib::MB << " MB\n";
        }
        std::cerr << std::endl;
        assert(false);                                                  //NOLINT
        std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));
        std::exit(EXIT_FAILURE);
    }
}

} // namespace detail
} // namespace xlib
