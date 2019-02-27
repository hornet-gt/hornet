/**
 * @internal
 * @brief Vectorized Bit Tree (Vec-Tree) interface
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
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
 *
 * @file
 */
#ifndef BITTREE_CUH
#define BITTREE_CUH

//#include "BasicTypes.hpp"                           //xlib::byte
#include <Host/Metaprogramming.hpp>                 //xlib::GeometricSerie
#include <utility>                                  //std::pair

namespace hornet {

/**
 * @brief **Vectorized Bit Tree**
 * @details Internal representation                                         <br>
 * [1011]                                   // root       --> internal      <br>
 * [0001] [0000] [1001] [0110]              //            --> internal      <br>
 * [....] [....] [....] [....] [....] ...   // last_level --> external      <br>
 *
 * @remark 1 means *block* available, 0 *block* used
 */
//TODO : Templatize with degree_t
template <typename degree_t>
class BitTree {
public:
    /**
     * @brief Default Costrustor
     * @details Build a empty *BitTree* with `blockarray_items` bits.
     *         It allocates a *BlockArray* for the HOST and another one for the
     *         DEVICE
     * @param[in] block_items Number of edges per *block*. Maximum number of
                              items that  that can fit in a single *block*
     * @param[in] blockarray_items Number of edges per *BlockArray*.
     *                          Maximum number of items that that can fit in a
     *                          single *BlockArray*
     * @pre BLOCK_ITEMS \f$\le\f$ BLOCKARRAY_ITEMS
     */
    BitTree(degree_t block_items, degree_t blockarray_items) noexcept;

    ~BitTree(void);

    /**
     * @brief Move constructor
     */
    BitTree(BitTree&& obj) noexcept;

    /**
     * @brief Default constructor
     */
    BitTree(void) noexcept = default;

    /**
     * @brief Copy constructor
     * @warning Internally replaced with the move constructor
     */
    BitTree(const BitTree& obj) noexcept;

    /**
     * @brief Assignment operator
     */
    BitTree& operator=(BitTree&& obj) noexcept;

    /**
     * @brief Assignment operator
     */
    BitTree& operator=(const BitTree& obj) noexcept;

    /**
     * @brief Insert a new *block*
     * @details Find the first empty *block* within the *BlockArray*
     * @return pointers to the *BlockArray*
     *         < `host_block_ptr`, `device_block_ptr` >
     */
    degree_t insert() noexcept;

    void remove(degree_t diff) noexcept;

    /**
     * @brief Size of the *BitTree*
     * @return number of used blocks within the *BlockArray*
     */
    degree_t size() const noexcept;

    /**
     * @brief Check if the *BitTree* is full
     * @return `true` if *BitTree* is full, `false` otherwise
     */
    bool full() const noexcept;

    /**
     * @brief Print BitTree internal representation
     */
    void print() const noexcept;

    /**
     * @brief Print BitTree statistics
     */
    void statistics() const noexcept;

    degree_t get_log_block_items() const noexcept;

    //--------------------------------------------------------------------------
private:
    using word_t = unsigned;
    static const unsigned   WORD_SIZE = sizeof(word_t) * 8;

    const degree_t _block_items	    { 0 };
    const degree_t _log_block_items	{ 0 };
    const degree_t _blockarray_items	{ 0 };
    const degree_t _num_blocks	    { 0 };
    const degree_t _num_levels	    { 0 };
    const degree_t _internal_bits	{ 0 };
    const degree_t _internal_words	{ 0 };
    const degree_t _external_words	{ 0 };
    const degree_t _num_words	    { 0 };
    const degree_t _total_bits	    { 0 };

    word_t* _array      { nullptr };
    word_t* _last_level { nullptr };
    long  _size       { 0 };

    template<typename Lambda>
    void parent_traverse(degree_t index, const Lambda& lambda) noexcept;
};

}

#include "BitTree.i.cuh"
#endif
