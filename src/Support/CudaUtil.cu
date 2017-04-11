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
        cudaDeviceReset();
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
