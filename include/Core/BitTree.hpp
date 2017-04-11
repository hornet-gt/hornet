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

#include "Support/Metaprogramming.hpp"
#include <utility>  //std::pair

namespace cu_stinger {

/**
 * @brief **Vectorized Bit Tree**
 * @details Internal representation                                         <br>
 * [1011]                                   // root       --> internal      <br>
 * [0001] [0000] [1001] [0110]              //            --> internal      <br>
 * [....] [....] [....] [....] [....] ...   // last_level --> external      <br>
 *
 * @remark 1 means *block* available, 0 *block* used
 * @tparam BLOCK_ITEMS Number of edges per *block*. Maximum number of items that
 *                     that can fit in a single *block*
 * @tparam BLOCKARRAY_ITEMS Number of edges per *BlockArray*.
 *                          Maximum number of items that that can fit in a
 *                          single *BlockArray*
 *
 * @pre BLOCK_ITEMS \f$\le\f$ BLOCKARRAY_ITEMS
 */
template<typename T, unsigned BLOCK_ITEMS, unsigned BLOCKARRAY_ITEMS>
class BitTree {
    using word_t = char;//unsigned;
    static_assert(BLOCK_ITEMS <= BLOCKARRAY_ITEMS, "BitTree Constrains");
public:
    /**
     * @brief Default Costrustor
     * @details Build a empty *BitTree* with `BLOCKARRAY_ITEMS` bits.
     *         It allocates a *BlockArray* for the HOST and other one for the
     *         DEVICE
     */
    BitTree() noexcept;

    /**
     * @brief Decostructor
     * @details Deallocate HOST and DEVICE *BlockArrays*
     */
    ~BitTree() noexcept;

    /**
     * @brief Free host *BlockArray* pointer
     */
    void free_host_ptr() noexcept;
    /**
     * @brief Insert a new *block*
     * @details Find the first empty *block* within the *BlockArray*
     * @return pair < host_block_ptr, device_block_ptr >
     */
    std::pair<T*, T*> insert() noexcept;

    /**
     * @brief Remove a *block*
     * @details Remove the *block* pointed by `to_delete` pointer
     * @param[in] to_delete pointer to the *block* to delete
     */
    void remove(T* to_delete) noexcept;

    /**
     * @brief Size of the *BitTree*
     * @return number of used blocks within the *BlockArray*
     */
    int size() const noexcept;

    /**
     * @brief Check if the *BitTree* is full
     * @return `true` if *BitTree* is full, `false` otherwise
     */
    bool is_full() const noexcept;

    /**
     * @brief Base address of the *BlockArray*
     * @return Pair < `host_block_ptr`, `device_block_ptr` > of the *BlockArray*
     */
    std::pair<T*, T*> base_address() const noexcept;

    /**
     * @brief Check if a particular *block* address belong to the actual
     *        *BlockArray*
     * @param[in] ptr pointer to check
     * @return `true` if ptr belong to *BitTree*  the actual *BlockArray*,
     *         `false` otherwise
     */
    bool belong_to(T* ptr) const noexcept;

    /**
     * @brief Print BitTree  internal representation
     */
    void print() const noexcept;

    /**
     * @brief Print BitTree statistics
     */
    void statistics() const noexcept;

    /**
     * @brief Move constructor
     */
    BitTree(BitTree&& obj) noexcept;

    /**
     * @brief Assignment operator
     */
    BitTree& operator=(BitTree&& obj) noexcept;
private:
    static const unsigned   WORD_SIZE = sizeof(word_t) * 8;
    static const auto      NUM_BLOCKS = BLOCKARRAY_ITEMS / BLOCK_ITEMS;
    static const auto      NUM_LEVELS = xlib::Max<xlib::CeilLog<NUM_BLOCKS,
                                                   WORD_SIZE>::value, 1>::value;
    //  WORD_SIZE^1 + WORD_SIZE^2 + ... + WORD_SIZE^(NUM_LEVELS - 1)
    using GS = typename xlib::GeometricSerie<WORD_SIZE, NUM_LEVELS - 1>;
    static const auto   INTERNAL_BITS = GS::value - 1;           //-1 : for root
    static const auto  INTERNAL_WORDS = xlib::CeilDiv<INTERNAL_BITS,
                                                      WORD_SIZE>::value;
    static const auto EXTERNAL_WORLDS = xlib::CeilDiv<NUM_BLOCKS,
                                                      WORD_SIZE>::value;
    static const auto       NUM_WORDS = INTERNAL_WORDS + EXTERNAL_WORLDS;
    static const auto      TOTAL_BITS = NUM_WORDS * WORD_SIZE;

    word_t  _array[NUM_WORDS];
    word_t* _last_level;
    T*      _h_ptr;
    T*      _d_ptr;
    size_t  _size;

    int remove_aux(T* to_delete) noexcept;

    template<typename Lambda>
    void parent_traverse(int index, const Lambda& lambda) noexcept;
};

} // namespace cu_stinger

#include "BitTree.i.hpp"
