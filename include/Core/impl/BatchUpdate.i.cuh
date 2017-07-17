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
#include "Support/Device/SafeCudaAPI.cuh"
#include "Support/Device/PrintExt.cuh"

namespace custinger {

inline BatchProperty::BatchProperty(const detail::BatchPropEnum& obj) noexcept :
    xlib::PropertyClass<detail::BatchPropEnum, BatchProperty>(obj) {}

//==============================================================================

inline BatchHost::BatchHost(vid_t* src_array, vid_t* dst_array, int batch_size)
                            noexcept :  _src_array(src_array),
                                        _dst_array(dst_array),
                                        _original_size(batch_size) {}


inline vid_t* BatchHost::original_src_ptr() const noexcept {
    return _src_array;
}

inline vid_t* BatchHost::original_dst_ptr() const noexcept {
    return _dst_array;
}

inline int BatchHost::original_size() const noexcept {
    return _original_size;
}

inline void BatchHost::print() const noexcept {
    xlib::printArray(_src_array, _original_size, "Source Array:\n");
    xlib::printArray(_dst_array, _original_size, "Destination Array:\n");
}

//==============================================================================

inline BatchDevice::BatchDevice(vid_t* src_array, vid_t* dst_array,
                                int batch_size) noexcept :
                                  BatchHost(src_array, dst_array, batch_size) {}

inline void BatchDevice::print() const noexcept {
    cu::printArray(_src_array, _original_size, "Source Array:\n");
    cu::printArray(_dst_array, _original_size, "Destination Array:\n");
}

//==============================================================================

inline BatchUpdate::BatchUpdate(vid_t max_batch_size,
                                size_t total_graph_edges,
                                vid_t num_vertices,
                                const BatchProperty& batch_prop) noexcept :
                                  _prop(batch_prop) {
    allocate(max_batch_size, total_graph_edges, num_vertices, batch_prop);
}

inline BatchUpdate::BatchUpdate(const BatchUpdate& batch_update) noexcept :
                                         _prop(batch_update._prop) {
    if (!_ready_for_device)
        ERROR("not ready for device")
}

inline BatchUpdate::BatchUpdate(const BatchHost& batch_host,
                                const BatchProperty& batch_prop,
                                size_t total_graph_edges,
                                vid_t num_vertices) noexcept :
                                    _prop(batch_prop) {
    allocate(batch_host.original_size(), total_graph_edges, num_vertices,
             batch_prop | batch_property::HOST);
}

inline BatchUpdate::BatchUpdate(const BatchDevice& batch_device,
                                const BatchProperty& batch_prop,
                                size_t total_graph_edges,
                                vid_t num_vertices) noexcept :
                                    _prop(batch_prop){

    allocate(batch_device.original_size(), total_graph_edges, num_vertices,
             batch_prop);
}

inline BatchUpdate::~BatchUpdate() noexcept {
    cuFree(_d_counts);
    if (_pinned)
        SAFE_CALL( cudaFreeHost(_batch_ptr) )
    else
        cuFree(_batch_ptr);
}

inline void BatchUpdate::allocate(vid_t max_batch_size,
                                  size_t total_graph_edges,
                                  vid_t num_vertices,
                                  const BatchProperty& batch_prop) noexcept {
    if (batch_prop != batch_property::LOW_MEMORY && total_graph_edges == 0)
        ERROR("!LOW_MEMORY && total_graph_edges == 0")

    if (batch_prop == batch_property::LOW_MEMORY)
        ;
    else if (batch_prop == batch_property::DELETE) {
        cuMalloc(_d_counts,       max_batch_size + 1,   //need
                 _d_unique,       max_batch_size,       //need
                 _d_degree_old,   max_batch_size + 1,
                 _d_degree_new,   max_batch_size + 1,
                 _d_tmp_sort_src, max_batch_size + 1,   //need
                 _d_tmp_sort_dst, max_batch_size + 1,   //need
                 _d_tmp,          total_graph_edges,
                 _d_ptrs_array,   max_batch_size + 1,
                 _d_flags,        total_graph_edges,    //need (V duplicates)
                 _d_inverse_pos,  num_vertices);        //need csr_wide
    }
    if (batch_prop == batch_property::HOST)
        allocate_batch(max_batch_size, batch_prop, true);
    else if (batch_prop == batch_property::COPY)
        allocate_batch(max_batch_size, batch_prop, false);
}

inline void BatchUpdate::allocate_batch(vid_t max_batch_size,
                                        const BatchProperty& batch_prop,
                                        bool pinned) noexcept {
    int  inverse = batch_prop == batch_property::GEN_INVERSE;
    _batch_pitch = xlib::upper_approx<512>(max_batch_size * sizeof(vid_t) *
                                           inverse);
    if (pinned)
        cuMallocHost(_batch_ptr, _batch_pitch * 2);
    else
        cuMalloc(_batch_ptr, _batch_pitch * 2);
    _d_src_array = reinterpret_cast<vid_t*>(_batch_ptr);
    _d_dst_array = reinterpret_cast<vid_t*>(_batch_ptr + _batch_pitch);
    _pinned      = pinned;
}

inline void BatchUpdate::bind(const BatchHost& batch_host) noexcept {
    size_t batch_size = batch_host.original_size();
    cuMemcpyToDevice(batch_host.original_src_ptr(), batch_size, _d_src_array);
    cuMemcpyToDevice(batch_host.original_dst_ptr(), batch_size, _d_dst_array);

    if (_prop == batch_property::GEN_INVERSE) {
        cuMemcpyDeviceToDevice(batch_host.original_src_ptr(), batch_size,
                               _d_dst_array + batch_size);
        cuMemcpyDeviceToDevice(batch_host.original_dst_ptr(), batch_size,
                               _d_src_array + batch_size);
        _batch_size = batch_size * 2;
    }
    else
        _batch_size = batch_size;
}

inline void BatchUpdate::bind(BatchDevice& batch_device) noexcept {
    int batch_size = batch_device.original_size();
    if (_batch_ptr == nullptr) {
        _d_src_array = batch_device.original_src_ptr();
        _d_dst_array = batch_device.original_dst_ptr();
    }
    if (_prop == batch_property::GEN_INVERSE) {
        cuMemcpyDeviceToDevice(batch_device.original_src_ptr(), batch_size,
                               _d_dst_array + batch_size);
        cuMemcpyDeviceToDevice(batch_device.original_dst_ptr(), batch_size,
                               _d_src_array + batch_size);
        _batch_size = batch_size * 2;
    }
    else
        _batch_size = batch_size;
}

//------------------------------------------------------------------------------

inline void BatchUpdate::change_ptrs(vid_t* d_src_array, vid_t* d_dst_array,
                                     int d_batch_size) noexcept {
    _d_src_array = d_src_array;
    _d_dst_array = d_dst_array;
    _batch_size  = d_batch_size;
    _ready_for_device = true;
}

inline void BatchUpdate::set_csr(const eoff_t* d_offsets, int offsets_size,
                                 eoff_t* d_inverse_pos) noexcept {
    _d_offsets     = d_offsets;
    _offsets_size  = offsets_size;
    _d_inverse_pos = d_inverse_pos;
}

inline bool BatchUpdate::ready_for_device() const noexcept {
    return _ready_for_device;
}

//------------------------------------------------------------------------------

HOST_DEVICE int BatchUpdate::size() const noexcept {
    return _batch_size;
}

HOST_DEVICE vid_t* BatchUpdate::src_ptr() const noexcept {
    return _d_src_array;
}

HOST_DEVICE vid_t* BatchUpdate::dst_ptr() const noexcept {
    return _d_dst_array;
}

HOST_DEVICE const eoff_t* BatchUpdate::offsets_ptr() const noexcept {
    assert(_d_offsets != nullptr);
    return _d_offsets;
}

HOST_DEVICE int BatchUpdate::offsets_size() const noexcept {
    assert(_d_offsets != 0);
    return _offsets_size;
}

#if defined(__NVCC__)

__device__ __forceinline__
vid_t BatchUpdate::src(int index) const noexcept {
    assert(index < _batch_size);
    return _d_src_array[index];
}

__device__ __forceinline__
vid_t BatchUpdate::dst(int index) const noexcept {
    assert(index < _batch_size);
    return _d_dst_array[index];
}

__device__ __forceinline__
int BatchUpdate::csr_src_pos(int index) const noexcept {
    assert(_d_inverse_pos != nullptr);
    assert(index < _batch_size);
    return _d_inverse_pos[index];
}

__device__ __forceinline__
vid_t BatchUpdate::csr_src(int index) const noexcept {
    assert(_d_offsets != nullptr);
    assert(index < _batch_size);
    return _d_offsets[index];
}

#endif

} // namespace custinger
