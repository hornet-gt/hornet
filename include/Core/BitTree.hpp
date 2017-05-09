/**
 * @internal
 * @brief Vec-Tree interface
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

#include "Support/Host/Metaprogramming.hpp"
#include <utility>  //std::pair

namespace custinger {

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
    using word_t = unsigned;
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
     * @return `true` if `ptr` belong to *BitTree*  the actual *BlockArray*,
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

} // namespace custinger

#include "BitTree.i.hpp"
