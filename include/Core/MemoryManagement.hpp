/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Core/BitTree.hpp"
#include <vector>

namespace cu_stinger {

using   edge_t = int;
using degree_t = int;

const degree_t      MIN_EDGES_PER_BLOCK = 2;
const degree_t MIN_EDGES_PER_BLOCKARRAY = 128;

template<typename T>
using Container = std::vector<T>;

//------------------------------------------------------------------------------

static_assert(xlib::IsPower2<MIN_EDGES_PER_BLOCK>::value      &&
              xlib::IsPower2<MIN_EDGES_PER_BLOCKARRAY>::value &&
              MIN_EDGES_PER_BLOCK <= MIN_EDGES_PER_BLOCKARRAY,
              "Memory Management Constrains");

template<degree_t BLOCK_ITEMS, degree_t BLOCKARRAY_ITEMS>
using BitTreeContainer = Container< BitTree<edge_t, BLOCK_ITEMS,
                                    BLOCKARRAY_ITEMS >>;
//------------------------------------------------------------------------------
///@cond
template<degree_t  LOW = MIN_EDGES_PER_BLOCK,
         degree_t HIGH = MIN_EDGES_PER_BLOCKARRAY,
         int     COUNT = 0,
         typename... T>
struct StructGen : StructGen<LOW * 2, HIGH, COUNT + 1, T...,
                             BitTreeContainer<LOW, HIGH>> {
    static_assert(xlib::IsPower2<LOW>::value && xlib::IsPower2<HIGH>::value &&
                  LOW <= HIGH, "LOW/HIGH must be power of 2 : 0 < LOW <= HIGH");
};

template<degree_t HIGH, int COUNT, typename... T>
struct StructGen<HIGH, HIGH, COUNT, T...> : StructGen<HIGH * 2, HIGH * 2,
                                                      COUNT + 1, T...,
                                                 BitTreeContainer<HIGH, HIGH>> {
    static_assert(xlib::IsPower2<HIGH>::value && HIGH != 0,
                  "HIGH must be power of 2 and HIGH != 0");
};


template<degree_t HIGH, typename... T>
struct StructGen<HIGH, HIGH, 29, T...> : std::tuple<T...,
                                                 BitTreeContainer<HIGH, HIGH>> {
    static_assert(xlib::IsPower2<HIGH>::value && HIGH != 0,
                  "HIGH must be power of 2 and HIGH != 0");
};

template<typename... T>
auto tuple_gen(std::tuple<T...>) -> typename std::tuple<T...> {
    return std::tuple<T...>();
}
///@endcond
//==============================================================================
//==============================================================================

/**
 * @brief The `MemoryManagement` class provides the infrastructure to organize
 *        the set of *BlockArrays*
 */
class MemoryManagement {
public:
    /**
     * @brief Default Costrustor
     * @details It creates an empty *BlockArray* container for each valid
     *          *block* size
     */
    MemoryManagement() noexcept;

    /**
     * @brief Free the host memory reserved for *BlockArrays*
     */
    void free_host_ptr() noexcept;

    /**
     * @brief Insert a new *block* of size
     *        \f$2^{\lceil \log_2(degree + 1) \rceil}\f$
     * @details If the actual *BlockArrays* of the correct size is full, it
     *          automatically allocates a new one.
     * @param[in] degree degree of the vertex to insert
     * @return Pair < `host_block_ptr`, `device_block_ptr` > of the
     *         corresponding *block*
     */
    std::pair<edge_t*, edge_t*> insert(degree_t degree) noexcept;

    /**
     * @brief Remove the *block* pointed by `ptr`
     * @param[in] ptr pointer to the *block* to remove
     * @param[in] degree degree of the vertex to remove
     * @warning the pointer must be previously inserted, otherwise an error is
     *          raised (debug mode)
     */
    void remove(edge_t* ptr, degree_t degree) noexcept;

    /**
     * @brief Number of *blocks*
     * @return the number of allocated *blocks*
     */
    int num_blocks() const noexcept;

    /**
     * @brief Various statistics about the space efficency
     * @details It provides the number of *BlockArrays*, the total number of
     *          allocated edges, the total number of used edges, the space
     *          efficiency (used/allocated), and the number of *blocks* for each
     *          level
     */
    void statistics() noexcept;

    /**
     * @brief Get the base pointer of a specific *BlockArrays*
     * @param[in] block_index index of the *BlockArrays*
     * @return Pair < `host_BlockArray_ptr`, `device_BlockArray_ptr` > of the
     *         corresponding *BlockArray*
     * @warning `block_index` must be
     *           \f$0 \le \text{block_index} < \text{total_block_arrays} \f$,
     *           otherwise an error is raised (debug mode)
     */
    std::pair<edge_t*, edge_t*> get_block_array_ptr(int block_index) noexcept;

private:
    int _num_blocks;
    decltype(tuple_gen(StructGen<>{})) bit_tree_set = tuple_gen(StructGen<>{});

    static constexpr int LIMIT = std::tuple_size<decltype(bit_tree_set)>::value;

    int find_bin(degree_t degree) const noexcept;

    template<template<int> class Lambda, typename... T>
    inline void traverse(int index, T&... args) noexcept;
};

} // namespace cu_stinger

#include "MemoryManagement.i.hpp"
