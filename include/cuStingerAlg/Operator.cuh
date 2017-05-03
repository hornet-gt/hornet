#pragma once

#include "Core/cuStingerTypes.cuh" //custinger::Vertex
//#include "Csr/CsrTypes.cuh"        //csr::Vertex

const int BLOCK_SIZE_OP1 = 256;

namespace cu_stinger_alg {
/////////////////
// C Style API //
/////////////////

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data);

template<void (*Operator)(custinger::vid_t, void*)>
void forAllnumV(void* optional_data);

template<void (*Operator)(custinger::eoff_t, void*)>
void forAllnumE(void* optional_data);

//------------------------------------------------------------------------------

template<void (*Operator)(custinger::Vertex, void*)>
void forAllVertices(void* optional_data);

template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllOutEdges(const custinger::eoff_t* out_offsets, void* optional_data);

//NOT IMPLEMENTED
template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllInEdges(const custinger::eoff_t* in_offsets, void* optional_data);
//------------------------------------------------------------------------------

template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllBatchEdges(const EdgeBatch& edge_batch, void* optional_data);

} // namespace cu_stinger_alg

#include "Operator.i.cuh"
