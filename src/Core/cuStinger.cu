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
#include "Core/cuStingerGlobalSpace.cuh"

namespace cu_stinger {

void cuStinger::initializeGlobal() noexcept {
    cuMalloc(_d_nodes, _nV);
    cuMemcpyToSymbol(_nV, d_nV);

    byte_t* h_ptrs[NUM_VTYPES];
    for (int i = 0; i < NUM_VTYPES; i++) {
        h_ptrs[i] = reinterpret_cast<byte_t*>(_d_nodes) +
                    _nV * VTYPE_SIZE_PS[i];
    }
    cuMemcpyToSymbol(h_ptrs, d_ptrs);
}

void cuStinger::print() noexcept {
    if (!_custinger_init)
        ERROR("cuStinger not initialized")
    if (sizeof(degree_t) == 4 && sizeof(id_t) == 4) {
        printKernel2<<<1, 1>>>();
        CHECK_CUDA_ERROR
    }
    else
        WARNING("Graph print is enable only with degree_t/id_t of size 4 bytes")
}

} // namespace cu_stinger
