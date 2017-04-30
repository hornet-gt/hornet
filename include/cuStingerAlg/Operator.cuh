
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

//==============================================================================
//==============================================================================
//==============================================================================
/////////////////
/// C++11 API ///
/////////////////

template<typename Lambda>
void forAll(size_t size, Lambda lambda);

template<typename Lambda>
void forAllVertices(Lambda lambda);

template<typename Lambda>
void forAllEdges(Lambda lambda);

template<typename Lambda>
void forAllnumV(Lambda lambda);

template<typename Lambda>
void forAllnumE(Lambda lambda);

} // namespace cu_stinger_alg

#include "Operator.i.cuh"
