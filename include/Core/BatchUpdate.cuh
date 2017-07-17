/**
 * @brief
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
 *
 * @file
 */
#pragma once

#if defined(__NVCC__)
    #include "Core/cuStingerTypes.cuh"
#endif
#include "Core/RawTypes.hpp"

namespace custinger {

namespace detail {
    enum class BatchPropEnum { GEN_INVERSE = 1, LOW_MEMORY = 2, CSR = 4,
                               CSR_WIDE = 8, DELETE = 16, INSERT = 32,
                               HOST = 64, REMOVE_DUPLICATE = 64, COPY = 128 };
} // namespace detail

class BatchProperty : public xlib::PropertyClass<detail::BatchPropEnum,
                                                 BatchProperty> {
public:
    explicit BatchProperty() noexcept = default;
    explicit BatchProperty(const detail::BatchPropEnum& obj) noexcept;
};

namespace batch_property {
    const BatchProperty GEN_INVERSE (detail::BatchPropEnum::GEN_INVERSE);
    const BatchProperty LOW_MEMORY  (detail::BatchPropEnum::LOW_MEMORY);
    const BatchProperty CSR         (detail::BatchPropEnum::CSR);
    const BatchProperty CSR_WIDE    (detail::BatchPropEnum::CSR_WIDE);
    const BatchProperty DELETE      (detail::BatchPropEnum::DELETE);
    const BatchProperty INSERT      (detail::BatchPropEnum::INSERT);
    const BatchProperty HOST        (detail::BatchPropEnum::HOST);
    const BatchProperty COPY        (detail::BatchPropEnum::COPY);
    const BatchProperty REMOVE_DUPLICATE
                                    (detail::BatchPropEnum::REMOVE_DUPLICATE);
}

//==============================================================================

class BatchHost {
public:
    explicit BatchHost(vid_t* src_array, vid_t* dst_array, int batch_size)
                       noexcept;

    vid_t* original_src_ptr() const noexcept;
    vid_t* original_dst_ptr() const noexcept;
    int    original_size()    const noexcept;

    virtual void print() const noexcept;

protected:
    vid_t* _src_array     { nullptr };
    vid_t* _dst_array     { nullptr };
    int    _original_size { 0 };
};

//==============================================================================

class BatchDevice : public BatchHost {
public:
    explicit BatchDevice(vid_t* src_array, vid_t* dst_array, int batch_size)
                         noexcept;

    void print() const noexcept override;
};

//==============================================================================

class BatchUpdate {
    friend class cuStinger;
public:
    explicit BatchUpdate(vid_t max_batch_size,
                         size_t total_graph_edges,
                         vid_t num_vertices,
                         const BatchProperty& batch_prop) noexcept;

    BatchUpdate(const BatchUpdate& batch_update) noexcept;

    explicit BatchUpdate(const BatchHost&     batch_host,
                         const BatchProperty& batch_prop      = BatchProperty(),
                         size_t               total_graph_edges = 0,
                         vid_t                num_vertices      = 0) noexcept;

    explicit BatchUpdate(const BatchDevice&   batch_host,
                         const BatchProperty& batch_prop      = BatchProperty(),
                         size_t               total_graph_edges = 0,
                         vid_t                num_vertices      = 0) noexcept;

    ~BatchUpdate() noexcept;

    void bind(const BatchHost& batch_host) noexcept;
    void bind(BatchDevice& batch_device) noexcept;

    bool ready_for_device() const noexcept;

    HOST_DEVICE int size() const noexcept;

    HOST_DEVICE int csr_size() const noexcept;

    HOST_DEVICE vid_t* src_ptr() const noexcept;

    HOST_DEVICE vid_t* dst_ptr() const noexcept;

    HOST_DEVICE const eoff_t* offsets_ptr() const noexcept;

    HOST_DEVICE int  offsets_size() const noexcept;

    void print() const noexcept;

#if defined(__NVCC__)
    __device__ __forceinline__
    vid_t src(int index) const noexcept;

    __device__ __forceinline__
    vid_t dst(int index) const noexcept;

    __device__ __forceinline__
    int csr_src_pos(int vertex_id) const noexcept;

    __device__ __forceinline__
    vid_t csr_src(int index) const noexcept;
#endif

private:
    const BatchProperty& _prop;

    vid_t* _d_src_array { nullptr };
    vid_t* _d_dst_array { nullptr };
    int    _batch_size  { 0 };

    const eoff_t* _d_offsets    { nullptr };
    int           _offsets_size { 0 };

    ///Batch delete tmp variables
    vid_t*    _d_unique       { nullptr };
    int*      _d_counts       { nullptr };
    degree_t* _d_degree_old   { nullptr };
    degree_t* _d_degree_new   { nullptr };
    byte_t*   *_d_ptrs_array  { nullptr };
    edge_t*   _d_tmp          { nullptr };
    bool*     _d_flags        { nullptr };
    eoff_t*   _d_inverse_pos  { nullptr };
    vid_t*    _d_tmp_sort_src { nullptr };
    vid_t*    _d_tmp_sort_dst { nullptr };

    byte_t*    _batch_ptr   { nullptr };
    size_t     _batch_pitch { 0 };  //bytes to the next field

    bool       _pinned           { false };
    bool       _ready_for_device { false };

    // 8 * BatchSize + 2E
    void allocate(vid_t max_batch_size, size_t total_graph_edges,
                  vid_t num_vertices, const BatchProperty& batch_prop) noexcept;

    void allocate_batch(vid_t max_batch_size, const BatchProperty& batch_prop,
                        bool pinned) noexcept;
    //--------------------------------------------------------------------------

    void change_ptrs(vid_t* d_src_array, vid_t* d_dst_array, int d_batch_size)
                     noexcept;

    void set_csr(const eoff_t* d_offsets, int offsets_size,
                 eoff_t* d_inverse_pos = nullptr) noexcept;
};

} // namespace custinger

#include "impl/BatchUpdate.i.cuh"
