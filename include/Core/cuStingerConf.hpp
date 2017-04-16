/*
 * @brief Example of cuStinger configuration
 * @file
 */
#pragma once

#include "Core/ConfigSupport.hpp"

namespace cu_stinger {
//------------------------------------------------------------------------------

using  id_t = int;      ///@brief vertex id type
using off_t = int;      ///@brief offset type

 ///@brief list of types for additional vertex data
using VertexTypes = TypeList<unsigned char>;

 ///@brief list of types for additional edge data
using   EdgeTypes = TypeList<uint64_t, float>;

///@brief minimum number of edges for a *block*
const size_t  MIN_EDGES_PER_BLOCK = 2;

///@brief number of edges for a *BlockArray*
const size_t EDGES_PER_BLOCKARRAY = 128;

//------------------------------------------------------------------------------

using degree_t = int;
using   edge_t = typename xlib::TupleConcat<TypeList<id_t>, EdgeTypes>::tuple;
using vertex_t = typename xlib::TupleConcat<
                       TypeList<edge_t*, degree_t, degree_t>, EdgeTypes>::tuple;

static_assert(xlib::IsPower2<MIN_EDGES_PER_BLOCK>::value  &&
              xlib::IsPower2<EDGES_PER_BLOCKARRAY>::value &&
              MIN_EDGES_PER_BLOCK <= EDGES_PER_BLOCKARRAY,
              "Memory Management Constrains");

using   VertexTypeSize =typename xlib::TupleToTypeSize<vertex_t>::sequence;
using     EdgeTypeSize =typename xlib::TupleToTypeSize<edge_t>::sequence;
using VertexTypeSizePS =typename xlib::PrefixSequence<VertexTypeSize>::sequence;
using   EdgeTypeSizePS =typename xlib::PrefixSequence<EdgeTypeSize>::sequence;

} // namespace cu_stinger
