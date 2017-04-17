/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
 */
#include "Support/CudaUtil.cuh"
#include "Support/Basic.hpp"
#include "Support/PrintExt.hpp"
#include <nvToolsExt.h>
#include <ostream>
#include <string>

namespace xlib {

void __getLastCudaError(const char* error_message, const char* file, int line,
                        const char* func_name) {
    __cudaErrorHandler(cudaGetLastError(), "", file, line, func_name);
}

void __safe_call(cudaError_t err, const char* file, int line,
                 const char* func_name) {
    __getLastCudaError("", file, line, func_name);
}

void __cudaErrorHandler(cudaError_t err, const char* error_message,
                        const char* file, int line, const char* func_name) {
    if (cudaSuccess != err) {
        std::cerr << Color::FG_RED << "\nCUDA error\n" << Color::FG_DEFAULT
                  << Emph::SET_UNDERLINE << file
                  << Emph::SET_RESET  << "(" << line << ")"
                  << " [ "
                  << Color::FG_L_CYAN << func_name  << Color::FG_DEFAULT
                  << " ] : " << error_message
                  << " -> " << cudaGetErrorString(err)
                  << "(" << static_cast<int>(err) << ")\n";
        if (static_cast<int>(err) == 2) {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            std::cerr << "\nActual allocated memory: " << std::setprecision(1)
                      << std::fixed << (total - free) / (1 << 20) << " MB\n";
        }
        std::cerr << std::endl;
        assert(false);                                                /*NOLINT*/
        std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));
        std::exit(EXIT_FAILURE);
    }
}

void memCheckCUDA(size_t Req) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (Req > free) {
        ERROR("Memory too low. Req: ", std::setprecision(1), std::fixed,
              static_cast<float>(Req) / (1 << 20), " MB, Available: ",
              static_cast<float>(free) / (1 << 20), " MB");
    }
}

bool memInfoCUDA(size_t Req) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    float percentage = static_cast<float>((Req >> 20) * 100) / (total >> 20);
    std::cout << std::endl << "[Device memory]" << std::endl
              << "     Total  " << (total >> 20) << " MB" << std::endl
              << " Requested  " << (Req >> 20)   << " MB"
              << std::setprecision(1) << "  (" << percentage << "%)"
              << std::endl;
    return free > Req;
}

int deviceProperty::NUM_OF_STREAMING_MULTIPROCESSOR = 0;

int deviceProperty::getNum_of_SMs() {
    if(NUM_OF_STREAMING_MULTIPROCESSOR == 0) {
        cudaDeviceProp devProperty;
        cudaGetDeviceProperties(&devProperty, 0);
        NUM_OF_STREAMING_MULTIPROCESSOR = devProperty.multiProcessorCount;
    }
    return NUM_OF_STREAMING_MULTIPROCESSOR;
}

namespace NVTX {
    /*void PushRange(std::string s, const int color) {
        nvtxEventAttributes_t eventAttrib = {};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color; //colors[color_id];
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = s.c_str();
        nvtxRangePushEx(&eventAttrib);
    }

    void PopRange() {
        nvtxRangePop();
    }*/
}

void deviceInfo() {
    cudaDeviceProp devProp;
    SAFE_CALL( cudaGetDeviceProperties(&devProp, 0) )

    std::cout << "\n     Graphic Card: " << devProp.name
              << " (cc: " << devProp.major << "."  << devProp.minor << ")\n"
              << "     # SM: "  << deviceProperty::getNum_of_SMs()
              << "    Threads per SM: " << devProp.maxThreadsPerMultiProcessor
              << "    Resident Threads: " << devProp.multiProcessorCount *
                                            devProp.maxThreadsPerMultiProcessor
              << "    Global Mem: "
              << (std::to_string(devProp.totalGlobalMem >> 20) + " MB\n")
              << std::endl;
}

} // namespace xlib
