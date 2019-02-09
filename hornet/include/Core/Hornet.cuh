/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
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
#pragma once
#include "Conf/Common.cuh"
#include "Conf/HornetConf.cuh"
#include "HornetDevice/HornetDevice.cuh"
#include "Core/HornetInitialize/HornetInit.cuh"
#include "BatchUpdate/BatchUpdate.cuh"
#include "MemoryManager/BlockArray/BlockArray.cuh"

namespace hornet {
namespace gpu {

template <typename, typename = EMPTY,
         typename = EMPTY, typename = DEGREE_T>
         class Hornet;

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
class Hornet<
    vid_t,
    TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>,
    degree_t> {

public:

    using HornetDeviceT = hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>;
    using EdgeAccessT = TypeList<degree_t, xlib::byte_t*, degree_t, degree_t>;
    using HInitT = hornet::HornetInit<
        vid_t,
        TypeList<VertexMetaTypes...>,
        TypeList<EdgeMetaTypes...>, degree_t>;
    using VertexTypes = TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>;

    using HostBlockArray = hornet::BlockArray<TypeList<vid_t, EdgeMetaTypes...>, DeviceType::HOST>;

private:

    static int _instance_count;

    vid_t    _nV { 0 };
    degree_t _nE { 0 };
    int      _id { 0 };

    SoAData<
        TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>,
        DeviceType::DEVICE> _vertex_data;

    BlockArrayManager<TypeList<vid_t, EdgeMetaTypes...>, DeviceType::DEVICE, degree_t> _ba_manager;

    void initialize(HInitT& h_init) noexcept;

    HornetDeviceT device(void);

    void reallocate_vertices(gpu::BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>& batch, const bool is_insert);

    void appendBatchEdges(gpu::BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>& batch);

public:

    Hornet(HInitT& h_init) noexcept;

    void insert(gpu::BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>& batch, bool removeBatchDuplicates = false, bool removeGraphDuplicates = false);

    void erase(gpu::BatchUpdate<vid_t, TypeList<EdgeMetaTypes...>, degree_t>& batch, bool removeBatchDuplicates = false);

    void print(void);

    degree_t nE(void) noexcept;
};

#define HORNET Hornet<vid_t,\
                      TypeList<VertexMetaTypes...>,\
                      TypeList<EdgeMetaTypes...>,\
                      degree_t>

}
}

#include "Core/HornetInitialize/HornetInitialize.i.cuh"
#include "Core/HornetOperations/HornetInsert.i.cuh"
