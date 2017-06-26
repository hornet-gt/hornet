/**
 * @internal
 * @brief Internal cuStinger types
 * @details Lowest level layer of the cuStinger programming model
 *          (hidden for the users)
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date June, 2017
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

static_assert(std::is_integral<vid_t>::value, "vid_t must be integral");
static_assert(std::is_integral<eoff_t>::value, "eoff_t must be integral");

static_assert(std::is_same<degree_t, int>::value ||
              std::is_same<degree_t, unsigned>::value ||
              std::is_same<degree_t, unsigned long long>::value,
              "degree_t type must allows atomicAdd operation");

//------------------------------------------------------------------------------

using      VTypeSize = typename xlib::TupleToTypeSize<vertex_t>::type;
using      ETypeSize = typename xlib::TupleToTypeSize<edge_t>::type;
using ExtraVTypeSize = typename xlib::TupleToTypeSize<VertexTypes>::type;
using ExtraETypeSize = typename xlib::TupleToTypeSize<EdgeTypes>::type;
using    VTypeSizePS = typename xlib::ExcPrefixSum<VTypeSize>::type;
using    ETypeSizePS = typename xlib::ExcPrefixSum<ETypeSize>::type;

///@internal @brief Array of all vertex field (type) sizes
extern const VTypeSize      VTYPE_SIZE;
///@internal @brief Array of all edge field (type) sizes
extern const ETypeSize      ETYPE_SIZE;
///@internal @brief Array of extra vertex field (type) sizes
extern const ExtraVTypeSize EXTRA_VTYPE_SIZE;
///@internal @brief Array of extra edge field (type) sizes
extern const ExtraETypeSize EXTRA_ETYPE_SIZE;

///@internal @brief Array of exclusive prefix-sum of all vertex field (type)
///                 sizes
extern const VTypeSizePS    VTYPE_SIZE_PS;
///@internal @brief Array of exclusive prefix-sum of all edge field (type) sizes
extern const ETypeSizePS    ETYPE_SIZE_PS;


///@internal @brief Number of all vertex fields (types)
const unsigned       NUM_VTYPES = std::tuple_size<vertex_t>::value;
///@internal @brief Number of all edge fields (types)
const unsigned       NUM_ETYPES = std::tuple_size<edge_t>::value;
///@internal @brief Number of extra vertex fields (types)
const unsigned NUM_EXTRA_VTYPES = std::tuple_size<VertexTypes>::value;
///@internal @brief Number of extra vertex fields (types)
const unsigned NUM_EXTRA_ETYPES = std::tuple_size<EdgeTypes>::value;
