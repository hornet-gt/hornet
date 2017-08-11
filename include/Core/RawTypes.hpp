/**
 * @internal
 * @brief Internal cuStinger types
 * @details Lowest level layer of the cuStinger programming model
 *          (hidden for the users)
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

#if defined(CSR_GRAPH)
    #include "Csr/RawTypes.hpp"
#else

#include "Support/Host/Basic.hpp"                    //xlib::byte_t
#include "Support/Host/Metaprogramming.hpp"          //xlib::TupleToTypeSize
#include "Core/MemoryManager/MemoryManagerConf.hpp"  //MIN_EDGES_PER_BLOCK
#include "Support/Host/Numeric.hpp"                  //xlib::roundup_pow2

namespace custinger {

using xlib::byte_t;
using degree_t = int;

template<typename... TArgs>
using TypeList = std::tuple<TArgs...>;

struct ALIGN(16) VertexBasicData {
    byte_t*  __restrict__ neighbor_ptr;
    degree_t              degree;

    HOST_DEVICE
    VertexBasicData(degree_t degree, byte_t* neighbor_ptr) :
        degree(degree), neighbor_ptr(neighbor_ptr) {}

    HOST_DEVICE degree_t limit() {
    #if defined(__CUDACC__)
        return ::max(static_cast<degree_t>(MIN_EDGES_PER_BLOCK),
                     PREFER_FASTER_UPDATE ? xlib::roundup_pow2(degree) :
                                            xlib::roundup_pow2(degree + 1));
    #else
        return std::max(static_cast<degree_t>(MIN_EDGES_PER_BLOCK),
                    PREFER_FASTER_UPDATE ? xlib::roundup_pow2(degree) :
                                           xlib::roundup_pow2(degree + 1));
    #endif
    }
};

///User configuration
//#include "configWeight.inc"
#include "config.inc"

using vertex_t = typename xlib::TupleConcat<TypeList<VertexBasicData>,
                                            VertexTypes>::type;

using   edge_t = typename xlib::TupleConcat<TypeList<vid_t>, EdgeTypes>::type;

#include "RawTypesUtil.hpp"

} // namespace custinger

#endif
