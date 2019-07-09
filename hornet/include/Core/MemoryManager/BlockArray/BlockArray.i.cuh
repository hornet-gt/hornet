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
namespace hornet {

#define BLOCK_ARRAY BlockArray<TypeList<Ts...>, device_t, degree_t>
#define B_A_MANAGER BlockArrayManager<TypeList<Ts...>, device_t, degree_t>

template<typename... Ts, DeviceType device_t, typename degree_t>
BLOCK_ARRAY::
BlockArray(const int block_items, const int blockarray_items) noexcept :
_edge_data(blockarray_items), _bit_tree(block_items, blockarray_items) {
}

template<typename... Ts, DeviceType device_t, typename degree_t>
BLOCK_ARRAY::
BlockArray(const BLOCK_ARRAY& other) noexcept :
_edge_data(other._edge_data), _bit_tree(other._bit_tree) {
}

template<typename... Ts, DeviceType device_t, typename degree_t>
BLOCK_ARRAY::
BlockArray(BLOCK_ARRAY&& other) noexcept :
_edge_data(std::move(other._edge_data)), _bit_tree(std::move(other._bit_tree)) {
}


template<typename... Ts, DeviceType device_t, typename degree_t>
xlib::byte_t *
BLOCK_ARRAY::
get_blockarray_ptr(void) noexcept {
    return reinterpret_cast<xlib::byte_t *>(_edge_data.get_soa_ptr().template get<0>());
}

template<typename... Ts, DeviceType device_t, typename degree_t>
int
BLOCK_ARRAY::
insert(void) noexcept {
    return _bit_tree.insert()<<_bit_tree.get_log_block_items();
}

template<typename... Ts, DeviceType device_t, typename degree_t>
void
BLOCK_ARRAY::
remove(int offset) noexcept {
    _bit_tree.remove(offset);
}

template<typename... Ts, DeviceType device_t, typename degree_t>
int
BLOCK_ARRAY::
capacity(void) noexcept {
    return _edge_data.get_num_items();
}

template<typename... Ts, DeviceType device_t, typename degree_t>
size_t
BLOCK_ARRAY::
mem_size(void) noexcept {
    return xlib::SizeSum<Ts...>::value * capacity();
}

template<typename... Ts, DeviceType device_t, typename degree_t>
bool
BLOCK_ARRAY::
full(void) noexcept {
    return _bit_tree.full();
}

template<typename... Ts, DeviceType device_t, typename degree_t>
CSoAData<TypeList<Ts...>, device_t>&
BLOCK_ARRAY::
get_soa_data(void) noexcept {
    return _edge_data;
}

//==============================================================================

template <typename degree_t>
int find_bin(const degree_t requested_degree) noexcept {
    return (requested_degree <= MIN_EDGES_PER_BLOCK ? 0 :
        xlib::ceil_log2(requested_degree) - xlib::Log2<MIN_EDGES_PER_BLOCK>::value);
}

template<typename... Ts, DeviceType device_t, typename degree_t>
B_A_MANAGER::
BlockArrayManager(const degree_t MaxEdgesPerBlockArray) noexcept :
_MaxEdgesPerBlockArray(1<<xlib::ceil_log2(MaxEdgesPerBlockArray)),
_largest_eb_size(1<<xlib::ceil_log2(MaxEdgesPerBlockArray)) {
}

template<typename... Ts, DeviceType device_t, typename degree_t>
template <DeviceType d_t>
B_A_MANAGER::
BlockArrayManager(const BlockArrayManager<TypeList<Ts...>, d_t, degree_t>& other) noexcept :
_ba_map(other._ba_map) {
}

template<typename... Ts, DeviceType device_t, typename degree_t>
template <DeviceType d_t>
B_A_MANAGER::
BlockArrayManager(BlockArrayManager<TypeList<Ts...>, d_t, degree_t>&& other) noexcept :
_ba_map(std::move(other._ba_map)) {
}

template<typename... Ts, DeviceType device_t, typename degree_t>
EdgeAccessData<degree_t>
B_A_MANAGER::
insert(const degree_t requested_degree) noexcept {
    int bin_index = find_bin(requested_degree);
    for (auto &ba : _ba_map[bin_index]) {
        if (!ba.second.full()) {
            degree_t offset = ba.second.insert();
            EdgeAccessData<degree_t> ea = {ba.second.get_blockarray_ptr(), offset, ba.second.capacity()};
            return ea;
        }
    }
    _largest_eb_size = std::max(1<<xlib::ceil_log2(requested_degree), _largest_eb_size);
    BLOCK_ARRAY new_block_array(
            1<<xlib::ceil_log2(requested_degree),
            std::max(1<<xlib::ceil_log2(requested_degree),
            _MaxEdgesPerBlockArray));
    degree_t offset = new_block_array.insert();
    EdgeAccessData<degree_t> ea = {new_block_array.get_blockarray_ptr(), offset, new_block_array.capacity()};

    auto block_ptr = new_block_array.get_blockarray_ptr();
    _ba_map[bin_index].insert(std::make_pair(block_ptr, std::move(new_block_array)));
    return ea;
}

template<typename... Ts, DeviceType device_t, typename degree_t>
void
B_A_MANAGER::
remove(
    degree_t       degree,
    xlib::byte_t * edge_block_ptr,
    degree_t       vertex_offset) noexcept {
    int bin_index = find_bin(degree);
    auto &ba = _ba_map[bin_index].at(edge_block_ptr);
    ba.remove(vertex_offset);
}

template<typename... Ts, DeviceType device_t, typename degree_t>
degree_t
B_A_MANAGER::
largest_edge_block_size(void) noexcept {
    return _largest_eb_size;
}

template<typename... Ts, DeviceType device_t, typename degree_t>
void
B_A_MANAGER::
removeAll(void) noexcept {
  for (auto &b : _ba_map) { b.clear(); }
}

#undef BLOCK_ARRAY
}
