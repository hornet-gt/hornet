#pragma once

namespace custinger_alg {
///////////////
// C++11 API //
///////////////
const int BLOCK_SIZE_OP2 = 256;

template<typename Operator>
void forAll(int num_items, Operator op);

template<typename Operator>
void forAllnumV(const custinger::cuStinger& custinger, Operator op);

template<typename Operator>
void forAllnumE(const custinger::cuStinger& custinger, Operator op);

//------------------------------------------------------------------------------

template<typename Operator>
void forAllVertices(const custinger::cuStinger& custinger, Operator op);

template<typename Operator>
void forAllEdges(const custinger::cuStinger& custinger, Operator op);

//------------------------------------------------------------------------------

template<typename Operator>
void forAllBatchEdges(Operator op);

template<typename Operator>
void forAllBatchVertices(Operator op);

} // namespace custinger_alg

#include "Operator++.i.cuh"
