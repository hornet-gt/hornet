/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
#include "Support/Host/Algorithm.hpp"
#include <iomanip>
#include <string>

namespace xlib {

//to update
template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool cuEqual(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B) {
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    CUDA_ERROR("Copy To Host");

    bool flag = xlib::equal<FAULT>(start_A, end_A, ArrayCMP);
    delete[] ArrayCMP;
    return flag;
}

//to update
template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool cuEqual(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
             bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type)) {

    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    CUDA_ERROR("Copy To Host");

    bool flag = xlib::equal<FAULT>(start_A, end_A, ArrayCMP, equalFunction);
    delete[] ArrayCMP;
    return flag;
}

//to update
template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool cuEqualSorted(iteratorA_t start_A, iteratorA_t end_A,
                   iteratorB_t start_B) {
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    CUDA_ERROR("Copy To Host");

    bool flag = xlib::equalSorted<FAULT>(start_A, end_A, ArrayCMP);
    delete[] ArrayCMP;
    return flag;
}

} // namespace xlib
