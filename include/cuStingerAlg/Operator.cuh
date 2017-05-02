#pragma once

#include "Core/cuStingerTypes.cuh" //cu_stinger::Vertex
//#include "Csr/CsrTypes.cuh"        //csr::Vertex

const int BLOCK_SIZE_OP1 = 256;

namespace cu_stinger_alg {
/////////////////
// C Style API //
/////////////////

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data);

template<void (*Operator)(cu_stinger::vid_t, void*)>
void forAllnumV(void* optional_data);

template<void (*Operator)(cu_stinger::eoff_t, void*)>
void forAllnumE(void* optional_data);

//------------------------------------------------------------------------------

template<void (*Operator)(cu_stinger::Vertex, void*)>
void forAllVertices(void* optional_data);

template<void (*Operator)(cu_stinger::Vertex, cu_stinger::Edge, void*)>
void forAllOutEdges(const cu_stinger::eoff_t* out_offsets, void* optional_data);

//NOT IMPLEMENTED
template<void (*Operator)(cu_stinger::Vertex, cu_stinger::Edge, void*)>
void forAllInEdges(const cu_stinger::eoff_t* in_offsets, void* optional_data);
//------------------------------------------------------------------------------

template<void (*Operator)(cu_stinger::Vertex, cu_stinger::Edge, void*)>
void forAllBatchEdges(const EdgeBatch& edge_batch, void* optional_data);

} // namespace cu_stinger_alg

#include "Operator.i.cuh"
