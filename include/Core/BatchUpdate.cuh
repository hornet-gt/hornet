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

/**
 * @brief Batch Property
 */
class BatchInit {
public:
    /**
     * @brief default costructor
     * @param[in] sort the edge batch is sorted in lexicographic order
     *            (source, destination)
     * @param[in] weighted_distr generate a batch by using a random weighted
     *            distribution based on the degree of the vertices
     * @param[in] print print the batch on the standard output
     */
    explicit BatchInit(const vid_t* src_array,  const vid_t* dst_array,
                       int batch_size) noexcept;

    /**
     * @brief Insert additional edge data
     * @param[in] edge_data list of edge data. The list must contains atleast
     *            the source and the destination arrays (vid_t type)
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *config.inc* file
     * @see ::insertVertexData
     */
    template<typename... TArgs>
    void insertEdgeData(TArgs... edge_data) noexcept;

    int size() const noexcept;

    const byte_t* edge_ptrs(int index) const noexcept;

private:
    const byte_t* _edge_ptrs[ NUM_ETYPES + 1 ] = {};
    const int     _batch_size { 0 };
};

/**
 * @brief Batch Property
 */
class Batch {
protected:
    /**
     * @brief default costructor
     * @param[in]
     * @param[in]
     * @param[in]
     */
    explicit Batch(const vid_t* src_array, const vid_t* dst_array,
                   int batch_size) noexcept;

    int size() const noexcept;

    const vid_t* src_array() const noexcept;
    const vid_t* dst_array() const noexcept;

    const vid_t* _src_array;
    const vid_t* _dst_array;
    const int    _batch_size;
};

class BatchHost : public Batch {
public:
    explicit BatchHost(const vid_t* src_array, const vid_t* dst_array,
                       int batch_size) noexcept;
    using Batch::size;
    using Batch::src_array;
    using Batch::dst_array;
};

class BatchDevice : public Batch {
public:
    explicit BatchDevice(const vid_t* src_array, const vid_t* dst_array,
                         int batch_size) noexcept;
    using Batch::size;
    using Batch::src_array;
    using Batch::dst_array;
};

//-----------------------------------------------------------------------------

/**
 * @brief Batch update class
 */
class BatchUpdate {
    friend class cuStinger;
public:
    /**
     * @brief default costructor
     * @param[in] batch_size number of edges of the batch
     */
    explicit BatchUpdate(const BatchInit& batch_init) noexcept;

    explicit BatchUpdate(size_t size) noexcept;

    //copy costructor to copy the batch to the kernel
    BatchUpdate(const BatchUpdate& obj) noexcept;

    ~BatchUpdate() noexcept;

    void insert(const BatchInit& batch_init) noexcept;

#if defined(__NVCC__)

    __host__ __device__ __forceinline__
    int size() const noexcept;

    __device__ __forceinline__
    vid_t src(int index) const noexcept;

    __device__ __forceinline__
    vid_t dst(int index) const noexcept;

    __device__ __forceinline__
    Edge edge(int index) const noexcept;

    template<int INDEX>
    __device__ __forceinline__
    typename std::tuple_element<INDEX, VertexTypes>::type
    field(int index) const noexcept;

    __device__ __forceinline__
    eoff_t* offsets_ptr() const noexcept;

    __device__ __forceinline__
    int offsets_size() const noexcept;

    __host__ __device__ __forceinline__
    vid_t* src_ptr() const noexcept;

    __host__ __device__ __forceinline__
    vid_t* dst_ptr() const noexcept;
#endif

private:
    byte_t*    _pinned_ptr    { nullptr };
    byte_t*    _d_edge_ptrs[ NUM_ETYPES + 1 ] = {};
    eoff_t*    _d_offsets     { nullptr };
    int        _batch_size    { 0 };
    int        _offsets_size  { 0 };
    const int  _batch_pitch   { 0 }; //number of edges to the next field
    const bool _enable_delete { true };
};

} // namespace custinger

#include "impl/BatchUpdate.i.cuh"
