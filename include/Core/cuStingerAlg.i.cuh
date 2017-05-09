/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
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
namespace custinger_alg {

inline StaticAlgorithm::StaticAlgorithm(custinger::cuStinger& custinger_)
                                        noexcept :
                                    custinger(custinger_),
                                    load_balacing(custinger_) {}

inline StaticAlgorithm::~StaticAlgorithm() noexcept {
    cuFree(_d_ptr);
}

template<typename T>
inline T* StaticAlgorithm::register_data(T& data) noexcept {
    if (_is_registered)
        ERROR("register_data() must be called only one times")
    _is_registered = true;
    _data_size     = sizeof(T);
    _h_ptr         = &data;
    SAFE_CALL( cudaMalloc(&_d_ptr, _data_size) )
    return reinterpret_cast<T*>(_d_ptr);
}

inline void StaticAlgorithm::syncHostWithDevice() noexcept {
    if (!_is_registered)
        ERROR("register_data() must be called before syncHostWithDevice()")
    SAFE_CALL( cudaMemcpy(_h_ptr, _d_ptr, _data_size, cudaMemcpyDeviceToHost) )
}

inline void StaticAlgorithm::syncDeviceWithHost() noexcept {
    if (!_is_registered)
        ERROR("register_data() must be called before syncDeviceWithHost()")
    SAFE_CALL( cudaMemcpy(_d_ptr, _h_ptr, _data_size, cudaMemcpyHostToDevice) )
}

//==============================================================================

template<typename T>
inline Allocate::Allocate(T*& pointer, size_t num_items) noexcept {
    cuMalloc(pointer, num_items);
    _pointer = pointer;
}

inline Allocate::~Allocate() noexcept {
    cuFree(_pointer);
}

} // namespace custinger_alg
