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
#ifndef BLOCK_ARRAY_CUH
#define BLOCK_ARRAY_CUH

#include "BitTree/BitTree.cuh"
#include "../../Conf/MemoryManagerConf.cuh" //EDGES_PER_BLOCKARRAY
#include "../../SoA/SoAData.cuh"
#include "../../Conf/HornetConf.cuh"
#include <array>
#include <unordered_map>

namespace hornet {

template <typename, DeviceType = DeviceType::DEVICE, typename = int> class BlockArray;
template <typename, DeviceType, typename> class BlockArrayManager;

template<typename... Ts, DeviceType device_t, typename degree_t>
class BlockArray<TypeList<Ts...>, device_t, degree_t> {

    template <typename, DeviceType, typename> friend class BlockArray;

    CSoAData<TypeList<Ts...>, device_t> _edge_data;
    BitTree<degree_t>                   _bit_tree;

    public:
    BlockArray(const int block_items, const int blockarray_items) noexcept;

    BlockArray(const BlockArray<TypeList<Ts...>, device_t, degree_t>& other) noexcept;

    BlockArray(BlockArray<TypeList<Ts...>, device_t, degree_t>&& other) noexcept;

    ~BlockArray(void) noexcept = default;

    xlib::byte_t * get_blockarray_ptr(void) noexcept;

    int insert(void) noexcept;

    void remove(int offset) noexcept;

    int capacity(void) noexcept;

    size_t mem_size(void) noexcept;

    bool full(void) noexcept;

    CSoAData<TypeList<Ts...>, device_t>& get_soa_data(void) noexcept;
};

template <typename degree_t>
struct EdgeAccessData {
    xlib::byte_t * edge_block_ptr;
    degree_t       vertex_offset;
    degree_t       edges_per_block;
};

template<typename... Ts, DeviceType device_t, typename degree_t>
class BlockArrayManager<TypeList<Ts...>, device_t, degree_t> {

    template <typename, DeviceType, typename> friend class BlockArrayManager;

    static constexpr unsigned LOG_DEGREE = sizeof(degree_t)*8;
    const degree_t _MaxEdgesPerBlockArray;
    degree_t _largest_eb_size;
    std::array<
        std::unordered_map<
            xlib::byte_t*,
            BlockArray<TypeList<Ts...>, device_t, degree_t>>,
    LOG_DEGREE> _ba_map;

    public:
    BlockArrayManager(const degree_t MaxEdgesPerBlockArray = EDGES_PER_BLOCKARRAY) noexcept;

    template <DeviceType d_t>
    BlockArrayManager(const BlockArrayManager<TypeList<Ts...>, d_t, degree_t>& other) noexcept;

    template <DeviceType d_t>
    BlockArrayManager(BlockArrayManager<TypeList<Ts...>, d_t, degree_t>&& other) noexcept;

    EdgeAccessData<degree_t> insert(const degree_t requested_degree) noexcept;

    void remove(
        degree_t       degree,
        xlib::byte_t * edge_block_ptr,
        degree_t       vertex_offset) noexcept;

    degree_t largest_edge_block_size(void) noexcept;

    void removeAll(void) noexcept;
};

}

#include "BlockArray.i.cuh"
#endif
