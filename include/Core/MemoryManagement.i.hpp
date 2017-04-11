/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
namespace cu_stinger {

namespace detail {

template<int INDEX>
struct MMInsert {
    template<typename T>
    static void op(T& bit_tree_set, std::pair<edge_t*, edge_t*>& ret) {
        auto& vect = std::get<INDEX>(bit_tree_set);
        for (auto& it : vect) {
            if (!it.is_full()) {
                ret = it.insert();
                return;
            }
        }
        const auto      BLOCK_ITEMS = MIN_EDGES_PER_BLOCK * (1 << INDEX);
        const auto BLOCKARRAY_ITEMS = BLOCK_ITEMS <= MIN_EDGES_PER_BLOCKARRAY ?
                                      MIN_EDGES_PER_BLOCKARRAY : BLOCK_ITEMS;
        vect.push_back(BitTree<edge_t, BLOCK_ITEMS, BLOCKARRAY_ITEMS>{});
        ret = vect.back().insert();
    }
};

template<int INDEX>
struct MMRemove {
    template<typename T>
    static void op(T& bit_tree_set, edge_t* ptr) {
        auto& vect = std::get<INDEX>(bit_tree_set);
        for (auto it = vect.begin(); it != vect.end(); it++) {
            //std::cout << INDEX << "\t" << ptr << "\t"
            //          << it->base_address().second << std::endl;
            if (it->belong_to(ptr)) {
                it->remove(ptr);
                if (it->size() == 0)
                    vect.erase(it);
                return;
            }
        }
        assert(false && "not found");
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
        const auto& vect = std::get<INDEX>(bit_tree_set);
        ret = vect.at(block_index).base_address();
    }
};

template<int INDEX>
struct MMStatistics {
    template<typename T>
    static void op(T& bit_tree_set, int& used_items, int& allocated_items) {
        const degree_t BLOCK_ITEMS = MIN_EDGES_PER_BLOCK * (1 << INDEX);
        const auto BLOCKARRAY_ITEMS = BLOCK_ITEMS <= MIN_EDGES_PER_BLOCKARRAY ?
                                      MIN_EDGES_PER_BLOCKARRAY : BLOCK_ITEMS;
        const auto& vect = std::get<INDEX>(bit_tree_set);
        int local_used_items = 0;
        for (const auto& it : vect)
            local_used_items += it.size();
        used_items      += local_used_items * BLOCK_ITEMS;
        allocated_items += vect.size() * BLOCKARRAY_ITEMS;
        std::cout << std::setw(4) << INDEX
                  << std::setw(15) << BLOCK_ITEMS
                  << std::setw(18) << BLOCKARRAY_ITEMS
                  << std::setw(16) << vect.size()
                  << std::setw(11) << local_used_items << "\n";
    }
};

} //namespace detail

//==============================================================================

inline MemoryManagement::MemoryManagement() noexcept : _num_blocks(0) {}

inline std::pair<edge_t*, edge_t*>
MemoryManagement::insert(degree_t degree) noexcept {
    _num_blocks++;
    int index = find_bin(degree);
    std::pair<edge_t*, edge_t*> ret;
    traverse<detail::MMInsert>(index, ret);
    return ret;
}

inline void MemoryManagement::remove(edge_t* ptr, degree_t degree) noexcept {
    _num_blocks--;
    int index = find_bin(degree);
    traverse<detail::MMRemove>(index, ptr);
}

inline void MemoryManagement::free_host_ptr() noexcept {
    for (int i = 0; i < LIMIT; i++)
        traverse<detail::MMFreeHost>(i);
}

inline std::pair<edge_t*, edge_t*>
MemoryManagement::get_block_array_ptr(int block_index) noexcept {
    assert(block_index >= 0);
    for (int i = 0; i < LIMIT; i++) {
        int vect_size = 0;
        traverse<detail::MMGetBlockPtr1>(i, vect_size);
        if (vect_size <= block_index) {
            std::pair<edge_t*, edge_t*> ret;
            traverse<detail::MMGetBlockPtr2>(i, block_index, ret);
            return ret;
        }
        block_index -= vect_size;
    }
    assert(false && "block_index out-of-bounds");
    return std::pair<edge_t*, edge_t*>(nullptr, nullptr);
}

//------------------------------------------------------------------------------

inline void MemoryManagement::statistics() noexcept {
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

    std::cout << "\n      N. blocks: " << _num_blocks
              << "\nAllocated Items: " << allocated_items
              << "\n     Used Items: " << used_items
              << "\nAllocated Space: " << (allocated_items >> 20) << " MB"
              << "\n     Used Space: " << (used_items >> 20) << " MB"
              << "\n     Efficiency: " << efficiency << " %\n" << std::endl;
}

inline int MemoryManagement::find_bin(degree_t degree) const noexcept {
    const unsigned LOG_EDGES_PER_BLOCK = xlib::Log2<MIN_EDGES_PER_BLOCK>::value;
    return degree < MIN_EDGES_PER_BLOCK ? 0 :
                     xlib::ceil_log2(degree + 1) - LOG_EDGES_PER_BLOCK;
}

inline int MemoryManagement::num_blocks() const noexcept {
    return _num_blocks;
}

//------------------------------------------------------------------------------

template<template<int> class Lambda, typename... T>
inline void MemoryManagement::traverse(int index, T&... args) noexcept {
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

} // namespace cu_stinger
