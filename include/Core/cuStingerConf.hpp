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
using VertexTypes = TypeList<id_t, float>;

 ///@brief list of types for additional edge data
using   EdgeTypes = TypeList<off_t, float, int>;

///@brief minimum number of edges for a *block*
const size_t  MIN_EDGES_PER_BLOCK = 2;

///@brief number of edges for a *BlockArray*
const size_t EDGES_PER_BLOCKARRAY = 128;

//------------------------------------------------------------------------------

using vertex_t = VertexTypes;
using   edge_t = EdgeTypes;
using degree_t = int;


static_assert(xlib::IsPower2<MIN_EDGES_PER_BLOCK>::value  &&
              xlib::IsPower2<EDGES_PER_BLOCKARRAY>::value &&
              MIN_EDGES_PER_BLOCK <= EDGES_PER_BLOCKARRAY,
              "Memory Management Constrains");
              
} // namespace cu_stinger
