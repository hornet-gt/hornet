/*
 * @brief Example of cuStinger configuration
 * @file
 */
#pragma once

#include <tuple>
#include "Support/Metaprogramming.hpp"

template<typename... TArgs>
using TypeList = std::tuple<TArgs...>;

namespace cu_stinger {
//------------------------------------------------------------------------------

#include "config.inc"

//------------------------------------------------------------------------------

using degree_t = int;
using   edge_t = typename xlib::TupleConcat<TypeList<id_t>, EdgeTypes>::type;
using vertex_t = typename xlib::TupleConcat<
                       TypeList<degree_t, degree_t, edge_t*>, EdgeTypes>::type;

static_assert(xlib::IsPower2<MIN_EDGES_PER_BLOCK>::value  &&
              xlib::IsPower2<EDGES_PER_BLOCKARRAY>::value &&
              MIN_EDGES_PER_BLOCK <= EDGES_PER_BLOCKARRAY,
              "Memory Management Constrains");

extern xlib::TupleToTypeSize<VertexTypes>::type  EXTRA_VTYPE_SIZE;
extern xlib::TupleToTypeSize<edge_t>::type       ETYPE_SIZE;
//extern xlib::PrefixSequence<decltype(VTYPE_SIZE)>::type VTYPE_SIZE_PS;
//extern xlib::PrefixSequence<decltype(ETYPE_SIZE)>::type ETYPE_SIZE_PS;

} // namespace cu_stinger
