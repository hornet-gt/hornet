#pragma once

#include "Core/cuStingerTypes.cuh"

namespace custinger_alg {
/////////////////
// C Style API //
/////////////////
const int BLOCK_SIZE_OP1 = 256;

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data) noexcept;

template<void (*Operator)(custinger::vid_t, void*)>
void forAllnumV(const custinger::cuStinger& custinger, void* optional_data)
                noexcept;

template<void (*Operator)(custinger::eoff_t, void*)>
void forAllnumE(const custinger::cuStinger& custinger, void* optional_data)
                noexcept;

//------------------------------------------------------------------------------

template<void (*Operator)(custinger::Vertex, void*)>
void forAllVertices(const custinger::cuStinger& custinger, void* optional_data)
                    noexcept;

template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllEdges(const custinger::cuStinger& custinger,
                 const custinger::eoff_t* out_offsets, void* optional_data)
                 noexcept;

//------------------------------------------------------------------------------
/*
template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllBatchEdges(const custinger::cuStinger& custinger,
                      const EdgeBatch& edge_batch, void* optional_data);
*/
} // namespace custinger_alg

#include "Operator.i.cuh"
