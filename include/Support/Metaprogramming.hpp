/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

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
template<unsigned... Is> struct seq{};

template<unsigned(*F)(unsigned), unsigned MAX, unsigned INDEX = 0,
         unsigned... Is>
struct gen_seq : gen_seq<F, MAX, INDEX + 1, Is..., F(INDEX)>{};

template<unsigned(*F)(unsigned), unsigned MAX, unsigned... Is>
struct gen_seq<F, MAX, MAX, Is...> : seq<Is...>{};

template<unsigned... Is>
constexpr std::array<unsigned const, sizeof...(Is)> array_gen(seq<Is...>) {
  return {{ Is... }};
}

template<unsigned(*F)(unsigned), unsigned MAX>
constexpr auto array_gen() -> decltype(array_gen(gen_seq<F, MAX>{}));

template<unsigned(*F)(unsigned), unsigned MAX>
constexpr auto array_gen() -> decltype(array_gen(gen_seq<F, MAX>{})) {
  return array_gen(gen_seq<F, MAX>{});
}

/*
constexpr int fun(int i) { return i * 3; };
constexpr auto table =  array_gen<fun, 4>();
f<table[0]>();
f<table[1]>();
*/

} // namespace xlib

#include "impl/Metaprogramming.i.hpp"
