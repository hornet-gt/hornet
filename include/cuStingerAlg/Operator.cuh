#pragma once

#include "Core/cuStingerTypes.cuh" //cu_stinger::Vertex
#include "Csr/CsrTypes.cuh"        //csr::Vertex

using cu_stinger::Vertex;
using cu_stinger::Edge;
const int BLOCK_SIZE_OP1 = 256;

namespace cu_stinger_alg {
/////////////////
// C Style API //
/////////////////

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data);

template<void (*Operator)(vid_t, void*)>
void forAllnumV(void* optional_data);

template<void (*Operator)(eoff_t, void*)>
void forAllnumE(void* optional_data);

//------------------------------------------------------------------------------

template<void (*Operator)(cu_stinger::Vertex, void*)>
void forAllVertices(void* optional_data);

template<void (*Operator)(cu_stinger::Vertex, cu_stinger::Edge, void*)>
void forAllOutEdges(const eoff_t* csr_offsets, void* optional_data);

//------------------------------------------------------------------------------

template<void (*Operator)(cu_stinger::Vertex, cu_stinger::Edge, void*)>
void forAllBatchEdges(void* optional_data);

} // namespace cu_stinger_alg

#include "Operator.i.cuh"
