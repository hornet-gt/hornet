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
#include <iomanip>  //std::setw

namespace custinger {

namespace detail {

template<int INDEX>
struct MMInsert {
    template<typename T>
    static void op(T& bit_tree_set, int& num_blockarrays,
                   std::pair<edge_t*, edge_t*>& ret) {
        auto& container = std::get<INDEX>(bit_tree_set);
        for (auto& it : container) {
            if (!it.is_full()) {
                ret = it.insert();
                return;
            }
        }
        const auto      BLOCK_ITEMS = MIN_EDGES_PER_BLOCK * (1 << INDEX);
        const auto BLOCKARRAY_ITEMS = BLOCK_ITEMS <= EDGES_PER_BLOCKARRAY ?
                                      EDGES_PER_BLOCKARRAY : BLOCK_ITEMS;
        container.push_back(BitTree<edge_t, BLOCK_ITEMS, BLOCKARRAY_ITEMS>{});
        ret = container.back().insert();
        num_blockarrays++;
    }
};

template<int INDEX>
struct MMRemove {
    template<typename T>
    static void op(T& bit_tree_set, int& num_blockarrays, edge_t* ptr) {
        auto& container = std::get<INDEX>(bit_tree_set);
        auto end_it = container.end();
        for (auto it = container.begin(); it != end_it; it++) {
            if (it->belong_to(ptr)) {
                it->remove(ptr);
                if (it->size() == 0) {
                    num_blockarrays--;
                    container.erase(it);
                }
                return;
            }
        }
        assert(false && "pointer not found");
    }
};

template<int INDEX>
struct MMFreeHost {
    template<typename T>
    static void op(T& bit_tree_set) {
        for (auto& it : std::get<INDEX>(bit_tree_set))
            it.free_host_ptr();
    }
};

template<int INDEX>
struct MMClear {
    template<typename T>
    static void op(T& bit_tree_set) {
        std::get<INDEX>(bit_tree_set).clear();
    }
};

template<int INDEX>
struct MMGetBlockPtr1 {
    template<typename T>
    static void op(T& bit_tree_set, int& size) {
        size = std::get<INDEX>(bit_tree_set).size();
    }
};

template<int INDEX>
struct MMGetBlockPtr2 {
    template<typename T>
    static void op(T& bit_tree_set, int block_index,
                   std::pair<edge_t*, edge_t*>& ret) {
        const auto& container = std::get<INDEX>(bit_tree_set);
        ret = container.at(block_index).base_address();
    }
};

template<int INDEX>
struct MMStatistics {
    template<typename T>
    static void op(T& bit_tree_set, int& used_items, int& allocated_items) {
        const degree_t  BLOCK_ITEMS = MIN_EDGES_PER_BLOCK * (1 << INDEX);
        const auto BLOCKARRAY_ITEMS = BLOCK_ITEMS <= EDGES_PER_BLOCKARRAY ?
                                      EDGES_PER_BLOCKARRAY : BLOCK_ITEMS;
        const auto& container = std::get<INDEX>(bit_tree_set);
        int local_used_items = 0;
        for (const auto& it : container)
            local_used_items += it.size();
        used_items      += local_used_items * BLOCK_ITEMS;
        allocated_items += container.size() * BLOCKARRAY_ITEMS;
        std::cout << std::setw(4) << INDEX
                  << std::setw(15) << BLOCK_ITEMS
                  << std::setw(18) << BLOCKARRAY_ITEMS
                  << std::setw(16) << container.size()
                  << std::setw(11) << local_used_items << "\n";
    }
};

} //namespace detail

//==============================================================================

inline std::pair<edge_t*, edge_t*>
MemoryManager::insert(degree_t degree) noexcept {
    int index = find_bin(degree);
    std::pair<edge_t*, edge_t*> ret;
    traverse<detail::MMInsert>(index, _num_blockarrays, ret);
    return ret;
}

inline void MemoryManager::remove(edge_t* ptr, degree_t degree) noexcept {
    if (ptr == nullptr)
        return;
    int index = find_bin(degree);
    traverse<detail::MMRemove>(index, _num_blockarrays, ptr);
}

inline std::pair<edge_t*, edge_t*>
MemoryManager::get_blockarray_ptr(int block_index) noexcept {
    assert(block_index >= 0 && block_index < _num_blockarrays);
    for (int i = 0; i < LIMIT; i++) {
        int vect_size = 0;
        traverse<detail::MMGetBlockPtr1>(i, vect_size);
        if (block_index < vect_size) {
            std::pair<edge_t*, edge_t*> ret;
            traverse<detail::MMGetBlockPtr2>(i, block_index, ret);
            return ret;
        }
        block_index -= vect_size;
    }
    assert(false && "block_index out-of-bounds");
    return std::pair<edge_t*, edge_t*>(nullptr, nullptr);
}

inline void MemoryManager::free_host_ptrs() noexcept {
    for (int i = 0; i < LIMIT; i++)
        traverse<detail::MMFreeHost>(i);
}

inline void MemoryManager::clear() noexcept {
    for (int i = 0; i < LIMIT; i++)
        traverse<detail::MMClear>(i);
}

//------------------------------------------------------------------------------

inline void MemoryManager::statistics() noexcept {
    std::cout << std::setw(5) << "IDX"
              << std::setw(14) << "BLOCKS_ITEMS"
              << std::setw(18) << "BLOCKARRAY_ITEMS"
              << std::setw(16) << "N. BLOCKARRAYS"
              << std::setw(11) << "N. BLOCKS" << "\n";
    int max_index = 0;
    for (int i = 0; i < LIMIT; i++) {
        int size;
        traverse<detail::MMGetBlockPtr1>(i, size);
        if (size > 0)
            max_index = i;
    }
    int allocated_items = 0, used_items = 0;
    for (int i = 0; i <= max_index; i++)
        traverse<detail::MMStatistics>(i, used_items, allocated_items);

    auto efficiency = xlib::per_cent(used_items, allocated_items);

    std::cout << "\n N. BlockArrays: " << _num_blockarrays
              << "\nAllocated Items: " << allocated_items
              << "\n     Used Items: " << used_items
              << "\nAllocated Space: " << (allocated_items >> 20) << " MB"
              << "\n     Used Space: " << (used_items >> 20) << " MB"
              << "\n     Efficiency: " << efficiency << " %\n" << std::endl;
}

inline int MemoryManager::find_bin(degree_t degree) const noexcept {
    const unsigned LOG_EDGES_PER_BLOCK = xlib::Log2<MIN_EDGES_PER_BLOCK>::value;
    return degree < MIN_EDGES_PER_BLOCK ? 0 :
             xlib::ceil_log2(degree + 1) - LOG_EDGES_PER_BLOCK;
    //return degree <= MIN_EDGES_PER_BLOCK ? 0 :
    //         xlib::ceil_log2(degree) - LOG_EDGES_PER_BLOCK;
}

inline int MemoryManager::num_blockarrays() const noexcept {
    return _num_blockarrays;
}

//------------------------------------------------------------------------------

template<template<int> class Lambda, typename... T>
inline void MemoryManager::traverse(int index, T&... args) noexcept {
    assert(index < LIMIT);
    switch (index) {
        case 0: Lambda<0>::op(bit_tree_set, args...); break;
        case 1: Lambda<1>::op(bit_tree_set, args...); break;
        case 2: Lambda<2>::op(bit_tree_set, args...); break;
        case 3: Lambda<3>::op(bit_tree_set, args...); break;
        case 4: Lambda<4>::op(bit_tree_set, args...); break;
        case 5: Lambda<5>::op(bit_tree_set, args...); break;
        case 6: Lambda<6>::op(bit_tree_set, args...); break;
        case 7: Lambda<7>::op(bit_tree_set, args...); break;
        case 8: Lambda<8>::op(bit_tree_set, args...); break;
        case 9: Lambda<9>::op(bit_tree_set, args...); break;
        case 10: Lambda<10>::op(bit_tree_set, args...); break;
        case 11: Lambda<11>::op(bit_tree_set, args...); break;
        case 12: Lambda<12>::op(bit_tree_set, args...); break;
        case 13: Lambda<13>::op(bit_tree_set, args...); break;
        case 14: Lambda<14>::op(bit_tree_set, args...); break;
        case 15: Lambda<15>::op(bit_tree_set, args...); break;
        case 16: Lambda<16>::op(bit_tree_set, args...); break;
        case 17: Lambda<17>::op(bit_tree_set, args...); break;
        case 18: Lambda<18>::op(bit_tree_set, args...); break;
        case 19: Lambda<19>::op(bit_tree_set, args...); break;
        case 20: Lambda<20>::op(bit_tree_set, args...); break;
        case 21: Lambda<21>::op(bit_tree_set, args...); break;
        case 22: Lambda<22>::op(bit_tree_set, args...); break;
        case 23: Lambda<23>::op(bit_tree_set, args...); break;
        case 24: Lambda<24>::op(bit_tree_set, args...); break;
        case 25: Lambda<25>::op(bit_tree_set, args...); break;
        case 26: Lambda<26>::op(bit_tree_set, args...); break;
        case 27: Lambda<27>::op(bit_tree_set, args...); break;
        case 28: Lambda<28>::op(bit_tree_set, args...); break;
        case 29: Lambda<29>::op(bit_tree_set, args...); break;
        default: ERROR("invalid index")
    }
}

} // namespace custinger
