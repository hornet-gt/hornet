#pragma once

#include "Core/cuStingerTypes.cuh" //cu_stinger::Vertex
#include "Csr/CsrTypes.cuh"        //csr::Vertex

using cu_stinger::Vertex;
using cu_stinger::Edge;
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
void forAllEdges(Operator op);

//------------------------------------------------------------------------------

template<typename Operator>
void forAllBatchEdges(Operator op);

} // namespace cu_stinger_alg

#include "Operator++.i.cuh"
