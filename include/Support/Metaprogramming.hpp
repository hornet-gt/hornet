/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Support/HostDevice.hpp"
#include <array>
#include <cstdint>

namespace xlib {

// ======================= COMPILE TIME numeric methods ========================
/**
 * @tparam A first parameter
 * @tparam B second parameter
 * @tparam ARGS additional parameters
 */
template<unsigned A, unsigned B, unsigned... ARGS> struct Max;
template<unsigned A, unsigned B, unsigned... ARGS> struct Min;
template<unsigned N, unsigned DIV>      struct CeilDiv;
template<uint64_t N, uint64_t DIV>      struct CeilDivUll;

/**
 * @brief compute
 * \f$ \[\left\lfloor {\frac{N}{{\textif{DIV}}} + 0.5} \right\rfloor \] \f$
 */
template<unsigned N, unsigned DIV>      struct RoundDiv;

template<unsigned N, unsigned MUL>      struct UpperApprox;
template<uint64_t N, uint64_t MUL>      struct UpperApproxUll;
template<unsigned N, unsigned MUL>      struct LowerApprox;
template<uint64_t N, uint64_t MUL>      struct LowerApproxUll;

template<uint64_t N>                    struct IsPower2;
template<unsigned N, unsigned EXP>      struct Pow;
template<unsigned N>                    struct RoundUpPow2;
template<uint64_t N>                    struct RoundUpPow2Ull;
template<unsigned N>                    struct RoundDownPow2;
template<uint64_t N>                    struct RoundDownPow2Ull;

template<unsigned N>                    struct Log2;
template<uint64_t N>                    struct Log2Ull;
template<unsigned N>                    struct CeilLog2;
template<uint64_t N>                    struct CeilLog2Ull;
template<unsigned N, unsigned BASE>     struct CeilLog;

template<unsigned N>                    struct Factorail;
template<unsigned N, unsigned K>        struct BinomialCoeff;
template<unsigned LOW, unsigned HIGH>   struct ProductSequence;
template<unsigned N, unsigned HIGH>     struct GeometricSerie;
//------------------------------------------------------------------------------

template<typename>
struct SeqDev;

template<unsigned... Is>
struct Seq {
    static constexpr unsigned value[] = { Is... };
    static constexpr unsigned    size = sizeof...(Is);
    constexpr unsigned operator[](int index) const { return value[index]; }
};
template<unsigned... Is>
constexpr unsigned Seq<Is...>::value[];
///@cond

template<unsigned... Is>
struct SeqDev<Seq<Is...>> {
    HOST_DEVICE unsigned operator[](int index) const {
         constexpr unsigned value[] = { Is... };
         return value[index];
    }
};

template<unsigned(*fun)(unsigned), unsigned MAX, unsigned INDEX = 0,
         unsigned... Is>
struct GenerateSeq : GenerateSeq<fun, MAX, INDEX + 1, Is..., fun(INDEX)>{};

template<unsigned(*fun)(unsigned), unsigned MAX, unsigned... Is>
struct GenerateSeq<fun, MAX, MAX, Is...>  {
    using type = Seq<Is...>;
};
/*
constexpr int fun(int i) { return i * 3; };
GenerateSeq<fun, 4>::type table;
f<table[0]>();
f<table[1]>();
*/

template<typename, typename>
struct TupleConcat;

template<typename... TArgs1, typename... TArgs2>
struct TupleConcat<std::tuple<TArgs1...>, std::tuple<TArgs2...>> {
    using type = std::tuple<TArgs1..., TArgs2...>;
};

//------------------------------------------------------------------------------

template<typename>
struct TupleToTypeSize;

template<typename... TArgs>
struct TupleToTypeSize<std::tuple<TArgs...>> {
   using type = Seq<sizeof(TArgs)...>;
};

//------------------------------------------------------------------------------

template<unsigned, typename, typename>
struct PrefixSumAux;

template<typename>
struct IncPrefixSum;

template<unsigned... Is>
struct IncPrefixSum<Seq<Is...>> :
    PrefixSumAux<sizeof...(Is), Seq<>, Seq<Is...>> {};

template<typename>
struct ExcPrefixSum;

template<unsigned... Is>
struct ExcPrefixSum<Seq<Is...>> :
    PrefixSumAux<sizeof...(Is) + 1, Seq<>, Seq<0, Is...>> {};

template<unsigned INDEX, unsigned I1, unsigned I2, unsigned... Is2>
struct PrefixSumAux<INDEX, Seq<>, Seq<I1, I2, Is2...>> :
       PrefixSumAux<INDEX - 1, Seq<I1, I1 + I2>,  Seq<I1 + I2, Is2...>> {};

template<unsigned INDEX, unsigned... Is1,
         unsigned I1, unsigned I2, unsigned... Is2>
struct PrefixSumAux<INDEX, Seq<Is1...>, Seq<I1, I2, Is2...>> :
   PrefixSumAux<INDEX - 1, Seq<Is1..., I1 + I2>,  Seq<I1 + I2, Is2...>> {};

template<unsigned... Is1, unsigned... Is2>
struct PrefixSumAux<1, Seq<Is1...>, Seq<Is2...>> {
    using type = Seq<Is1...>;
};
//@endcond

} // namespace xlib

#include "impl/Metaprogramming.i.hpp"
