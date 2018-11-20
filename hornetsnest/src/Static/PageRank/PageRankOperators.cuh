/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
 *   ogreen@gatech.edu
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
#pragma once

namespace hornets_nest {

struct InitOperator {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(vid_t src) {
        hd_prdata().curr_pr[src]   = 0.0f;
    	hd_prdata().abs_diff[src]  = 0.0f;
    	hd_prdata().prev_pr[src]   = 1.0f / static_cast<float>(hd_prdata().nV);
    	*hd_prdata().reduction_out = 0;
    }
};

//------------------------------------------------------------------------------

struct ResetCurr {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(vid_t src) {
    	hd_prdata().curr_pr[src]     = 0.0f;
    	*(hd_prdata().reduction_out) = 0;
    }
};

//------------------------------------------------------------------------------

struct ComputeContribuitionPerVertex {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(Vertex& vertex_src) {
    	auto degree = vertex_src.degree();
        auto    src = vertex_src.id();
		hd_prdata().contri[src] = degree == 0 ? 0.0f :
                                  hd_prdata().prev_pr[src] / degree;
    }
};

//------------------------------------------------------------------------------

struct AddContribuitionsPush {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(Vertex& src, Edge& edge) {
        atomicAdd(hd_prdata().curr_pr + edge.dst_id(),
                  hd_prdata().contri[src.id()]);
    }
};

//------------------------------------------------------------------------------

struct AddContribuitionsPull {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(Vertex& src, Edge& edge) {
        atomicAdd(hd_prdata().curr_pr+ src.id(),
                  hd_prdata().contri[edge.dst_id()]);
    }
};

//------------------------------------------------------------------------------

struct DampAndDiffAndCopy {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(vid_t src) {
    	hd_prdata().curr_pr[src]  = hd_prdata().normalized_damp +
                                   hd_prdata().damp * hd_prdata().curr_pr[src];

    	hd_prdata().abs_diff[src] = fabsf(hd_prdata().curr_pr[src] -
                                         hd_prdata().prev_pr[src]);
    	hd_prdata().prev_pr[src]  = hd_prdata().curr_pr[src];
    }
};

//------------------------------------------------------------------------------

struct Sum {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(vid_t src) {
        atomicAdd(hd_prdata().reduction_out, hd_prdata().abs_diff[src]);
    }
};

//------------------------------------------------------------------------------

struct SumPr {
    HostDeviceVar<PrData> hd_prdata;

    OPERATOR(vid_t src) {
        atomicAdd(hd_prdata().reduction_out, hd_prdata().prev_pr[src] );
    }
};

//------------------------------------------------------------------------------

struct SetIds {
    vid_t* ids;

    OPERATOR(vid_t src) {
        ids[src] = src;
    }
};

} // hornets_nest namespace
