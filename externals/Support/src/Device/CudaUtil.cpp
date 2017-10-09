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
#include "Device/CudaUtil.cuh"
#include "Host/Basic.hpp"       //xlib::MB
#include "Host/PrintExt.hpp"    //Color
#include <iomanip>

namespace xlib {

void __getLastCudaError(const char* file, int line, const char* func_name) {
    __cudaErrorHandler(cudaGetLastError(), "", file, line, func_name);
}

void __safe_call(cudaError_t error, const char* file, int line,
                 const char* func_name) {
    __cudaErrorHandler(error, "", file, line, func_name);
}

void __cudaErrorHandler(cudaError_t error, const char* error_message,
                        const char* file, int line, const char* func_name) {
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

int DeviceProperty::NUM_OF_STREAMING_MULTIPROCESSOR = 0;                //NOLINT

int DeviceProperty::num_SM() {
    if (NUM_OF_STREAMING_MULTIPROCESSOR == 0) {
        cudaDeviceProp prop;
        SAFE_CALL( cudaGetDeviceProperties(&prop, 0) );
        NUM_OF_STREAMING_MULTIPROCESSOR = prop.multiProcessorCount;
    }
    return NUM_OF_STREAMING_MULTIPROCESSOR;
}


void device_info(int device_id) {
    xlib::IosFlagSaver tmp1;
    xlib::ThousandSep  tmp2;

   /* std::cout << "\n     Graphic Card: " << prop.name
              << " (CC: " << prop.major << "."  << prop.minor << ")\n"
              << "     # SM: "  << prop.multiProcessorCount
              << "    Threads per SM: " << prop.maxThreadsPerMultiProcessor
              << "    Resident Threads: " << prop.multiProcessorCount *
                                            prop.maxThreadsPerMultiProcessor
              << "    Global Mem: "
              << (std::to_string(prop.totalGlobalMem / xlib::MB) + " MB")
              << std::endl;*/

    int dev_peak_clock;
    cudaDeviceGetAttribute(&dev_peak_clock, cudaDevAttrClockRate, device_id);
    cudaDeviceProp prop;
    SAFE_CALL( cudaGetDeviceProperties(&prop, device_id) )

    auto smem = std::to_string(prop.sharedMemPerMultiprocessor /xlib::KB)
                       + " KB";
    auto gmem = xlib::format(prop.totalGlobalMem /xlib::MB) + " MB";
    auto smem_thread = prop.sharedMemPerMultiprocessor /
                        prop.maxThreadsPerMultiProcessor;
    auto  thread_regs = prop.regsPerMultiprocessor /
                        prop.maxThreadsPerMultiProcessor;
    auto      l2cache = std::to_string(prop.l2CacheSize / xlib::MB) + " MB";

    std::cout << std::boolalpha << std::setprecision(1) << std::left
              << std::fixed << "\n"
    << "    GPU: " << prop.name
    << "   CC: "   << prop.major << "." << prop.minor
    << "   #SM: "  << prop.multiProcessorCount
    << "   @"      << (prop.clockRate / 1000) << "/"
                   << (dev_peak_clock / 1000) << " MHz\n"
    << "           Threads (SM): " << std::setw(13)
                                   << prop.maxThreadsPerMultiProcessor
    << "Registers (thread): "      << thread_regs << "\n"
    << "        Shared Mem (SM): " << std::setw(12) << smem
    << "Shared Mem (thread): "     << smem_thread << " B\n"
    << "             Global Mem: " << std::setw(15) << gmem
    << "Resident threads: "        << prop.multiProcessorCount *
                                      prop.maxThreadsPerMultiProcessor
    << "\n"
    << "               L2 cache: "  << std::setw(15) << l2cache
    << "L1 caching (l/g): "        << prop.localL1CacheSupported << "/"
                                   << prop.globalL1CacheSupported
    << "\n" << std::endl;
}

} // namespace xlib
