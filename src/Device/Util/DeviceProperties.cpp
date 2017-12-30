/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
 * @version v1.4
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
#include "Device/Util/DeviceProperties.cuh"     //xlib::DeviceProperty
#include "Device/Util/SafeCudaAPI.cuh"          //SAFE_CALL
#include <cuda_runtime.h>                       //cudaGetDeviceCount

namespace xlib {

int  DeviceProperty::_num_sm[DeviceProperty::MAX_GPUS]        = {};
int  DeviceProperty::_smem_per_SM[DeviceProperty::MAX_GPUS]   = {};
int  DeviceProperty::_rblock_per_SM[DeviceProperty::MAX_GPUS] = {};
int  DeviceProperty::_num_gpus  = 0;
bool DeviceProperty::_init_flag = false;

void DeviceProperty::_init() noexcept {
    _init_flag = true;
    SAFE_CALL( cudaGetDeviceCount(&_num_gpus) )
    for (int i = 0; i < _num_gpus; i++) {
        cudaDeviceProp devive_prop;
        SAFE_CALL( cudaGetDeviceProperties(&devive_prop, i) )
        _num_sm[i]        = devive_prop.multiProcessorCount;
        _smem_per_SM[i]   = devive_prop.sharedMemPerMultiprocessor;
        _rblock_per_SM[i] = (devive_prop.major >= 5) ? 32 : 16;
    }
}

int DeviceProperty::num_SM() noexcept {
    if (!_init_flag)
        _init();
    return _num_sm[(_num_gpus == 1) ? 0 : cuGetDevice()];
}

int DeviceProperty::smem_per_SM()noexcept {
    if (!_init_flag)
        _init();
    return _smem_per_SM[(_num_gpus == 1) ? 0 : cuGetDevice()];
}

int DeviceProperty::resident_blocks_per_SM() noexcept {
    if (!_init_flag)
        _init();
    return _rblock_per_SM[(_num_gpus == 1) ? 0 : cuGetDevice()];
}

//------------------------------------------------------------------------------

int DeviceProperty::resident_threads() noexcept {
    return num_SM() * xlib::THREADS_PER_SM;
}

int DeviceProperty::resident_warps() noexcept {
    return num_SM() * (xlib::THREADS_PER_SM / xlib::WARP_SIZE);
}

int DeviceProperty::resident_blocks(int block_size) noexcept {
    auto size = xlib::upper_approx<xlib::WARP_SIZE>(block_size);
    return num_SM() * (xlib::THREADS_PER_SM / static_cast<unsigned>(size));
}

} // namespace xlib
