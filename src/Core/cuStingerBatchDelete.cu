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
#if !defined(CSR_GRAPH)

#include "Core/cuStinger.hpp"
#include "Core/Kernels/BatchDeleteKernels.cuh"
#include "Support/Device/CubWrapper.cuh"        //xlib::CubSortPairs2
#include "Support/Device/PrintExt.cuh"          //cu::printArray
#include <type_traits>                          //std::conditional
//#define BATCH_DELETE_DEBUG
//#define CHECK_DUPLICATE

namespace custinger {

static const unsigned BLOCK_SIZE = 256;

void cuStinger::deleteEdgeBatch(BatchUpdate& batch_update,
                                BatchProperty batch_prop) noexcept {
    /*auto _d_tmp_sort_src = batch_update._d_tmp_sort_src;
    auto _d_tmp_sort_dst = batch_update._d_tmp_sort_dst;
    auto        _d_flags = batch_update.__d_flags;
    auto       _d_counts = batch_update.__d_counts;
    auto       _d_unique = batch_update.__d_unique;
    auto   _d_degree_old = batch_update.__d_degree_old;
    auto   _d_degree_new = batch_update.__d_degree_new;
    auto  _d_inverse_pos = batch_update.__d_inverse_pos;
    auto   _d_ptrs_array = batch_update.__d_ptrs_array;
    auto          _d_tmp = batch_update._d_tmp;*/
    size_t   batch_size = batch_update.size();
    vid_t*  d_batch_src = batch_update.src_ptr();
    vid_t*  d_batch_dst = batch_update.dst_ptr();

#if defined(BATCH_DELETE_DEBUG)
    cu::printArray(d_batch_src, batch_size, "INPUT:\n");
    cu::printArray(d_batch_dst, batch_size);
#endif
#if defined(CHECK_DUPLICATE)
    auto src_tmp = new int[batch_size];
    auto dst_tmp = new int[batch_size];
    auto d_array = new std::pair<int, int>[batch_size];
    cuMemcpyToHost(d_batch_src, batch_size, src_tmp);
    cuMemcpyToHost(d_batch_dst, batch_size, dst_tmp);
    for (int i = 0; i < batch_size; i++)
        d_array[i] = { src_tmp[i], dst_tmp[i] };
    std::sort(d_array, d_array + batch_size);
    auto it = std::unique(d_array, d_array + batch_size);
    if (std::distance(d_array, it) != batch_size)
        ERROR("The batch contains duplicates")
    delete[] src_tmp;
    delete[] dst_tmp;
    delete[] d_array;
#endif
    //==========================================================================
    ///////////////////
    // SORT + UNIQUE //
    ///////////////////
    if (batch_prop == batch_property::REMOVE_DUPLICATE) {
        xlib::CubSortPairs2<vid_t, vid_t> sort_cub(d_batch_src, d_batch_dst,
                                                   batch_size, _d_tmp_sort_src,
                                                   _d_tmp_sort_dst, _nV, _nV);
        sort_cub.run(); // batch sorting
        //----------------------------------------------------------------------
        xlib::CubSelectFlagged<vid_t> select_src(d_batch_src, batch_size,
                                                 _d_flags);
        xlib::CubSelectFlagged<vid_t> select_dst(d_batch_dst, batch_size,
                                                 _d_flags);

        markUniqueKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
            (d_batch_src, d_batch_dst, batch_size, _d_flags);
        CHECK_CUDA_ERROR

        batch_size = select_src.run();
        select_dst.run();
    }
    else {
        xlib::CubSortByKey<vid_t, vid_t> sort_cub(d_batch_src, d_batch_dst,
                                                  batch_size, _d_tmp_sort_src,
                                                  _d_tmp_sort_dst, _nV);
        sort_cub.run();
        d_batch_src = _d_tmp_sort_src;
        d_batch_dst = _d_tmp_sort_dst;
    }
#if defined(BATCH_DELETE_DEBUG)
    cu::printArray(d_batch_src, batch_size, "Sort (+ Unique):\n");
    cu::printArray(d_batch_dst, batch_size);
#endif
    //==========================================================================
    batch_update.change_ptrs(d_batch_src, d_batch_dst, batch_size);
    //==========================================================================
    ///////////////
    // RUNLENGTH //
    ///////////////
    xlib::CubRunLengthEncode<vid_t> runlength_cub(d_batch_src, batch_size,
                                                  _d_unique, _d_counts);

    int num_uniques = runlength_cub.run();

    collectOldDegreeKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (device_side(), _d_unique, num_uniques, _d_degree_old, _d_inverse_pos);
    CHECK_CUDA_ERROR

#if defined(BATCH_DELETE_DEBUG)
    std::cout << "num_uniques " << num_uniques << std::endl;
    cu::printArray(_d_degree_old, num_uniques, "_d_degree_old\n");
#endif
    xlib::CubExclusiveSum<degree_t> prefixsum1(_d_degree_old, num_uniques + 1);
    prefixsum1.run();
    //==========================================================================

    degree_t total_degree_old;      //get the total collected degree
    cuMemcpyToHost(_d_degree_old + num_uniques, total_degree_old);
    cuMemset0xFF(_d_flags, total_degree_old);

    deleteEdgesKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
        (device_side(), batch_update, _d_degree_old, _d_inverse_pos, _d_flags);
    CHECK_CUDA_ERROR

    collectDataKernel   //modify also the vertices degree
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (device_side(), _d_unique, _d_counts, num_uniques,
         _d_degree_new, _d_ptrs_array);
    CHECK_CUDA_ERROR

#if defined(BATCH_DELETE_DEBUG)
    cu::printArray(_d_degree_new, num_uniques, "_d_degree_new\n");
#endif

    xlib::CubExclusiveSum<degree_t> prefixsum2(_d_degree_new, num_uniques + 1);
    prefixsum2.run();

#if defined(BATCH_DELETE_DEBUG)
    cu::printArray(_d_degree_old, num_uniques + 1, "_d_degree_old_prefix\n");
    cu::printArray(_d_degree_new, num_uniques + 1, "_d_degree_new_prefix\n");
#endif
    degree_t total_degree_new;                //get the total
    cuMemcpyToHost(_d_degree_new + num_uniques, total_degree_new);
    //==========================================================================

    const int SMEM = xlib::SMemPerBlock<BLOCK_SIZE, degree_t>::value;
    int num_blocks = xlib::ceil_div<SMEM>(total_degree_old);

    moveDataKernel1<BLOCK_SIZE, SMEM> <<< num_blocks, BLOCK_SIZE >>>
        (_d_degree_old, num_uniques + 1, _d_ptrs_array, _d_tmp);
    CHECK_CUDA_ERROR

    using tmp_t = typename std::conditional<NUM_EXTRA_ETYPES == 1,
                                            int2, vid_t>::type;
    xlib::CubSelectFlagged<tmp_t>
             select(reinterpret_cast<tmp_t*>(_d_tmp), total_degree_old, _d_flags);
    int tmp_size_new = select.run();
    assert(total_degree_new == tmp_size_new);

    if (total_degree_new > 0) {
        num_blocks = xlib::ceil_div<SMEM>(total_degree_new);
        moveDataKernel2<BLOCK_SIZE, SMEM> <<< num_blocks, BLOCK_SIZE >>>
            (_d_degree_new, num_uniques + 1, _d_tmp, _d_ptrs_array);
        CHECK_CUDA_ERROR
    }
    //==========================================================================

    if (batch_prop == batch_property::CSR) {
        xlib::CubExclusiveSum<int> prefixsum3(_d_counts, num_uniques + 1);
        prefixsum3.run();
        batch_update.set_csr(_d_degree_old, num_uniques, _d_inverse_pos);
    }
    else if (batch_prop == batch_property::CSR_WIDE) {
        cuMemset0x00(_d_inverse_pos, _nV + 1);
        scatterDegreeKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
            (_d_unique, _d_counts, num_uniques, _d_inverse_pos);

        xlib::CubExclusiveSum<int> prefixsum3(_d_inverse_pos, _nV + 1);
        prefixsum3.run();
        batch_update.set_csr(_d_inverse_pos, _nV);
    }
}

} // namespace custinger

#undef BATCH_DELETE_DEBUG
#undef CHECK_DUPLICATE
#endif
