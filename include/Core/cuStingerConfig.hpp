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

#include "../config.inc"

//------------------------------------------------------------------------------

using degree_t = int;
using   edge_t = typename xlib::TupleConcat<TypeList<id_t>, EdgeTypes>::type;
using vertex_t = typename xlib::TupleConcat<
                                       TypeList<degree_t, degree_t, edge_t*>,
                                       VertexTypes>::type;

static_assert(xlib::IsPower2<MIN_EDGES_PER_BLOCK>::value  &&
              xlib::IsPower2<EDGES_PER_BLOCKARRAY>::value &&
              MIN_EDGES_PER_BLOCK <= EDGES_PER_BLOCKARRAY,
              "Memory Management Constrains");

using VextexSizes = typename xlib::TupleToTypeSize<vertex_t>::type;

extern const xlib::TupleToTypeSize<vertex_t>::type            VTYPE_SIZE;
extern const xlib::TupleToTypeSize<VertexTypes>::type         EXTRA_VTYPE_SIZE;
extern const xlib::TupleToTypeSize<edge_t>::type              ETYPE_SIZE;

extern const xlib::ExcPrefixSum<VextexSizes>::type VTYPE_SIZE_PS;
//extern xlib::PrefixSequence<decltype(ETYPE_SIZE)>::type ETYPE_SIZE_PS;

const unsigned NUM_EXTRA_VTYPES = std::tuple_size<VertexTypes>::value;
const unsigned NUM_EXTRA_ETYPES = std::tuple_size<EdgeTypes>::value;
const unsigned       NUM_VTYPES = std::tuple_size<vertex_t>::value;
const unsigned       NUM_ETYPES = std::tuple_size<edge_t>::value;

} // namespace cu_stinger
