#pragma once

#include "Core/cuStingerTypes.cuh" //custinger::Vertex
#include "Csr/CsrTypes.cuh"        //csr::Vertex

using custinger::Vertex;
using custinger::Edge;
const int BLOCK_SIZE_OP2 = 256;

namespace cu_stinger_alg {
///////////////
// C++11 API //
///////////////

template<typename Operator>
void forAll(int num_items, Operator op);

template<typename Operator>
void forAllnumV(Operator op);

template<typename Operator>
void forAllnumE(Operator op);

//------------------------------------------------------------------------------

template<typename Operator>
void forAllVertices(Operator op);

template<typename Operator>
void forAllOutEdges(Operator op);

template<typename Operator>
void forAllInEdges(Operator op);

//------------------------------------------------------------------------------

template<typename Operator>
void forAllBatchEdges(Operator op);

} // namespace cu_stinger_alg

#include "Operator++.i.cuh"
