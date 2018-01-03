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
#include "Kernels/BatchInsertKernels.cuh"
#include <Device/Util/PrintExt.cuh>      //cu::printArray

//#define DEBUG_INSERT

namespace hornets_nest {
namespace gpu {

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::insertEdgeBatch(BatchUpdate& batch_update) noexcept {
    const unsigned BLOCK_SIZE = 128;
    int num_uniques = batch_preprocessing(batch_update, true);
    //==========================================================================
    size_t  batch_size = batch_update.size();
    vid_t* d_batch_src = batch_update.src_ptr();
    vid_t* d_batch_dst = batch_update.dst_ptr();
                                       //is_insert, get_old_degree
    fixInternalRepresentation(num_uniques, true, true);
    //==========================================================================
    //////////////////////
    // OPERATE ON BATCH //
    //////////////////////
    cub_prefixsum.run(_d_counts, num_uniques + 1);

    if (_is_sorted) { // IN_PLACE SORTING !! (may be slow)
        //Merge sort
        mergeAdjListKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
            (device_side(), _d_degree_tmp, _d_unique,
             _d_counts, num_uniques, d_batch_dst);
        CHECK_CUDA_ERROR
    }
    else {  // BULK COPY BATCH INTO HORNET
        int smem = xlib::DeviceProperty::smem_per_block(BLOCK_SIZE);
        int num_blocks = xlib::ceil_div(batch_size, smem);
#if defined(DEBUG_INSERT)
        cu::printArray(_d_degree_tmp, num_uniques, "_d_degree_tmp:\n");
#endif
        bulkCopyAdjLists<BLOCK_SIZE>  <<< num_blocks, BLOCK_SIZE >>>
            (device_side(), _d_counts, num_uniques + 1,
             d_batch_dst, _d_unique, _d_degree_tmp);
        CHECK_CUDA_ERROR
    }

    if (_batch_prop == batch_property::CSR)
        build_batch_csr(batch_update, num_uniques, false);
}

} // namespace gpu
} // namespace hornets_nest

#undef DEBUG_INSERT
