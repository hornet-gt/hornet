#pragma once

namespace cu_stinger_alg {

template<typename Operator, typename T, typename... TArgs>
void forAll(T* d_array, int num_items, TArgs... args);

template<typename Operator, typename T, typename... TArgs>
void forAllnumV(T* d_array, TArgs... args);

template<typename Operator, typename T, typename... TArgs>
void forAllnumE(T* d_array, TArgs... args);

//------------------------------------------------------------------------------

template<typename Operator, typename... TArgs>
void forAllVertices(TArgs... optional_data);

template<typename Operator, typename... TArgs>
void forAllEdges(TArgs... optional_data);

//------------------------------------------------------------------------------

template<typename Operator, typename... TArgs>
void forAllBatchEdges(TArgs... optional_data);

template<typename Operator, typename T, typename... TArgs>
void forAllTraverseEdges(Queue<T> queue, TArgs... optional_data);

template<typename Operator, typename T, typename... TArgs>
void forAllTraverseEdges(T* d_array, int num_items, TArgs... optional_data);

} // namespace cu_stinger_alg

#include "Operator.i.cuh"
