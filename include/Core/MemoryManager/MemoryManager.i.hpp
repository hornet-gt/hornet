/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
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
#include <iomanip>  //std::setw

namespace custinger {

inline MemoryManager::MemoryManager() noexcept {
#if !defined(B_PLUS_TREE)
    const auto  LOW = MIN_EDGES_PER_BLOCK;
    const auto HIGH = EDGES_PER_BLOCKARRAY;
    for (size_t size = LOW, i = 0; size <= HIGH; size *= 2, i++)
        bit_tree_set[i].reserve(512);
#endif
}

inline std::pair<byte_t*, byte_t*>
MemoryManager::insert(degree_t degree) noexcept {
    const auto HIGH = EDGES_PER_BLOCKARRAY;
    assert(degree > 0);
    int index = find_bin(degree);
    _num_inserted_edges += degree;

    auto& container = bit_tree_set[index];
    if (container.size() == 0) {
#if defined(B_PLUS_TREE)
        BitTree bit_tree(MIN_EDGES_PER_BLOCK * (1 << index), HIGH);
        container.insert(std::make_pair(bit_tree.base_address().second,
                                        bit_tree));
#else
        container.push_back(BitTree(MIN_EDGES_PER_BLOCK * (1 << index), HIGH));
#endif
        _num_blockarrays++;
    }
    for (auto& it : container) {
        if (!it.SC_MACRO full())
            return it.SC_MACRO insert();
    }
    _num_blockarrays++;
    auto      block_items = MIN_EDGES_PER_BLOCK * (1 << index);
    auto blockarray_items = block_items <= EDGES_PER_BLOCKARRAY ?
                            EDGES_PER_BLOCKARRAY : block_items;
#if defined(B_PLUS_TREE)
    BitTree bit_tree(block_items, blockarray_items);
    auto ret = bit_tree.insert();
    container.insert(std::make_pair(bit_tree.base_address().second, bit_tree));
    return ret;
#else
    container.push_back(BitTree(block_items, blockarray_items));
    return container.back().insert();
#endif
}

inline void MemoryManager::remove(void* device_ptr, degree_t degree) noexcept {
    assert(degree > 0);
    if (device_ptr == nullptr)
        return;
    int index = find_bin(degree);
    _num_inserted_edges -= degree;

    auto& container = bit_tree_set[index];
#if defined(B_PLUS_TREE)
    byte_t* low_address = reinterpret_cast<byte_t*>(device_ptr)
                          - EDGES_PER_BLOCKARRAY;
    const auto& it = container.upper_bound(low_address);
    assert(it != container.end());
    it->second.remove(device_ptr);
    if (it->second.size() == 0) {  //shrink
        _num_blockarrays--;
        container.erase(it);
    }
#else
    const auto& end_it = container.end();
    for (auto it = container.begin(); it != end_it; it++) {
        if (it->belong_to(device_ptr)) {
            it->remove(device_ptr);
            if (it->size() == 0) {  //shrink
                _num_blockarrays--;
                container.erase(it);
            }
            return;
        }
    }
    assert(false && "pointer not found");
#endif
}

inline std::pair<byte_t*, byte_t*>
MemoryManager::get_blockarray_ptr(int blockarray_index) noexcept {
    assert(blockarray_index >= 0 && blockarray_index < _num_blockarrays);
    for (int i = 0; i < MM_LOG_LIMIT; i++) {
        int container_size = bit_tree_set[i].size();

        if (blockarray_index < container_size) {
            const auto& it = std::next(bit_tree_set[i].begin(),
                                       blockarray_index);
            return it->SC_MACRO base_address();
        }
        blockarray_index -= container_size;
    }
    assert(false && "blockarray_index out-of-bounds");
    return std::pair<byte_t*, byte_t*>(nullptr, nullptr);
}

inline void MemoryManager::free_host_ptrs() noexcept {
    for (int i = 0; i < MM_LOG_LIMIT; i++) {
        for (auto& it : bit_tree_set[i])
            it.SC_MACRO free_host_ptr();
    }
}

inline void MemoryManager::clear() noexcept {
    for (int i = 0; i < MM_LOG_LIMIT; i++)
        bit_tree_set[i].clear();
}

//------------------------------------------------------------------------------

inline void MemoryManager::statistics() noexcept {
    std::cout << std::setw(5)  << "IDX"
              << std::setw(14) << "BLOCKS_ITEMS"
              << std::setw(18) << "BLOCKARRAY_ITEMS"
              << std::setw(16) << "N. BLOCKARRAYS"
              << std::setw(11) << "N. BLOCKS" << "\n";
    int max_index = 0;
    for (int i = 0; i < MM_LOG_LIMIT; i++) {
        if (bit_tree_set[i].size() > 0)
            max_index = i;
    }
    int allocated_items = 0, used_items = 0;
    for (int index = 0; index <= max_index; index++) {
        const degree_t  block_items = MIN_EDGES_PER_BLOCK * (1 << index);
        const auto blockarray_items = block_items <= EDGES_PER_BLOCKARRAY ?
                                      EDGES_PER_BLOCKARRAY : block_items;
        const auto& container = bit_tree_set[index];
        int local_used_items = 0;
        for (const auto& it : container)
            local_used_items += it.SC_MACRO size();
        used_items      += local_used_items * block_items;
        allocated_items += container.size() * blockarray_items;

        std::cout << std::setw(4)  << index
                  << std::setw(15) << block_items
                  << std::setw(18) << blockarray_items
                  << std::setw(16) << container.size()
                  << std::setw(11) << local_used_items << "\n";
    }
    auto efficiency1 = xlib::per_cent(_num_inserted_edges, used_items);
    auto efficiency2 = xlib::per_cent(used_items, allocated_items);

    std::cout << "\n         N. BlockArrays: " << xlib::format(_num_blockarrays)
              << "\n        Allocated Items: " << xlib::format(allocated_items)
              << "\n             Used Items: " << xlib::format(used_items)
              << "\n        Allocated Space: " << (allocated_items >> 20)
                                               << " MB"
              << "\n             Used Space: " << (used_items >> 20) << " MB"
              << "\n  (Internal) Efficiency: " << xlib::format(efficiency1, 1)
                                               << " %"
              << "\n  (External) Efficiency: " << xlib::format(efficiency2, 1)
              << " %\n" << std::endl;
}

inline int MemoryManager::find_bin(degree_t degree) const noexcept {
    const unsigned LOG_EDGES_PER_BLOCK = xlib::Log2<MIN_EDGES_PER_BLOCK>::value;
    return PREFER_FASTER_UPDATE ?
        (degree < MIN_EDGES_PER_BLOCK ? 0 :
             xlib::ceil_log2(degree + 1) - LOG_EDGES_PER_BLOCK) :
        (degree <= MIN_EDGES_PER_BLOCK ? 0 :
            xlib::ceil_log2(degree) - LOG_EDGES_PER_BLOCK);
}

inline int MemoryManager::num_blockarrays() const noexcept {
    return _num_blockarrays;
}

} // namespace custinger
