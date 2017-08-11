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
#include "Core/cuStinger.hpp"
#include "Core/Kernels/cuStingerKernels.cuh"
#include "Core/cuStingerTypes.cuh"              //VertexBasicData
#include "Support/Device/BinarySearchLB.cuh"    //xlib::binarySearchLB
#include "Support/Device/CubWrapper.cuh"        //CubSortByValue
#include "Support/Device/Definition.cuh"        //xlib::SMemPerBlock
#include "Support/Device/PrintExt.cuh"          //xlib::SMemPerBlock

namespace custinger {

void cuStinger::print() noexcept {
    if (sizeof(degree_t) == 4 && sizeof(vid_t) == 4) {
        printKernel<<<1, 1>>>(device_side());
        CHECK_CUDA_ERROR
    }
    else {
        WARNING("Graph print is enabled only with degree_t/vid_t of size"
                " 4 bytes")
    }
}

/*
 * !!!!! 4E + 2V
 */
void cuStinger::transpose() noexcept {
    const unsigned BLOCK_SIZE = 256;
    mem_manager.clear();

    eoff_t* d_csr_offsets, *d_counts_out;
    vid_t*  d_coo_src, *d_coo_dst, *d_coo_src_out, *d_coo_dst_out,*d_unique_out;
    cuMalloc(d_csr_offsets, _nV + 1);
    cuMalloc(d_coo_src, _nE);
    cuMalloc(d_coo_dst, _nE);
    cuMalloc(d_coo_src_out, _nE);
    cuMalloc(d_coo_dst_out, _nE);
    cuMalloc(d_counts_out, _nV + 1);
    cuMalloc(d_unique_out, _nV);
    cuMemcpyToDeviceAsync(_csr_offsets, _nV + 1, d_csr_offsets);
    cuMemcpyToDeviceAsync(_csr_edges, _nE, d_coo_dst);
    cuMemcpyToDeviceAsync(0, d_counts_out + _nV);

    CSRtoCOOKernel<BLOCK_SIZE>
        <<< xlib::ceil_div(_nV, BLOCK_SIZE), BLOCK_SIZE >>>
        (d_csr_offsets, _nV, d_coo_dst);

    xlib::CubSortByKey<vid_t, vid_t>(d_coo_dst, d_coo_src, _nE,
                                     d_coo_dst_out, d_coo_src_out, _nV - 1);
    xlib::CubRunLengthEncode<vid_t, eoff_t>(d_coo_dst_out, _nE,
                                            d_unique_out, d_counts_out);
    xlib::CubExclusiveSum<eoff_t>(d_counts_out, _nV + 1);

    _csr_offsets = new eoff_t[_nV + 1];
    _csr_edges   = new vid_t[_nV + 1];
    cuMemcpyToHostAsync(d_counts_out, _nV + 1,
                        const_cast<eoff_t*>(_csr_offsets));
    cuMemcpyToHostAsync(d_coo_src_out, _nE, const_cast<vid_t*>(_csr_edges));
    _internal_csr_data = true;
    cuFree(d_coo_dst, d_coo_src, d_counts_out, d_unique_out,
           d_coo_src_out, d_counts_out);
    initialize();
}

void cuStinger::check_sorted_adjs() const noexcept {
    checkSortedKernel <<< xlib::ceil_div(_nV, 256), 256 >>> (device_side());
}
//------------------------------------------------------------------------------

void cuStinger::build_device_degrees() noexcept {
    cuMalloc(_d_degrees, _nV);
    buildDegreeKernel <<< xlib::ceil_div(_nV, 256), 256 >>>
        (device_side(), _d_degrees);
}

vid_t cuStinger::max_degree_id() noexcept {
    if (max_degree_data.first == -1) {
        xlib::CubArgMax<degree_t> arg_max(_d_degrees, _nV);
        max_degree_data = arg_max.run();
    }
    return max_degree_data.first;
}

vid_t cuStinger::max_degree() noexcept {
    if (max_degree_data.first == -1) {
        xlib::CubArgMax<degree_t> arg_max(_d_degrees, _nV);
        max_degree_data = arg_max.run();
    }
    return max_degree_data.second;
}

} // namespace custinger
