/**
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
 */
#include "Host/Numeric.hpp"     //xlib::factorial

namespace xlib {

template<unsigned N, unsigned DIV>
struct CeilDiv {
    static_assert(DIV != 0, "division by zero in integer arithmetic");
    static const unsigned value = N == 0 ? 0 : 1 + ((N - 1) / DIV);
};

template<uint64_t N, uint64_t DIV>
struct CeilDivUll {
    static_assert(DIV != 0, "division by zero in integer arithmetic");
    static const uint64_t value = N == 0 ? 0 : 1 + ((N - 1) / DIV);
};

template<unsigned N, unsigned DIV>
struct RoundDiv {
    static const unsigned value = (N + (DIV / 2)) / DIV;
};

template<unsigned N, unsigned MUL>
struct LowerApprox {
    static const unsigned value = (N / MUL) * MUL;
};
template<uint64_t N, uint64_t MUL>
struct LowerApproxUll {
    static const uint64_t value = (N / MUL) * MUL;
};

template<unsigned N, unsigned MUL>
struct UpperApprox {
    static const unsigned value = CeilDiv<N, MUL>::value * MUL;
};
template<uint64_t N, uint64_t MUL>
struct UpperApproxUll {
    static const uint64_t value = CeilDivUll<N, MUL>::value * MUL;
};

template<unsigned N, unsigned EXP>
struct Pow {
    static const unsigned value = Pow<N, EXP - 1>::value * N;
};
template<unsigned N>
struct Pow<N, 0> {
    static const unsigned value = 1;
};

template<unsigned N>
struct RoundUpPow2 {
private:
    static const unsigned V = N - 1;
public:
    static const unsigned value = (V | (V >> 1) | (V >> 2) |
                                  (V >> 4) | (V >> 8) | (V >> 16)) + 1;
};
template<uint64_t N>
struct RoundUpPow2Ull {
private:
    static const uint64_t V = N - 1;
public:
    static const uint64_t value = (V | (V >> 1) | (V >> 2) |
                                  (V >> 4) | (V >> 8) | (V >> 16)) + 1;
};

template<unsigned N>
struct RoundDownPow2 {
private:
    static const unsigned V = RoundUpPow2<N>::value;
public:
    static const unsigned value = V == N ? N : V >> 1;
};

template<uint64_t N>
struct RoundDownPow2Ull {
private:
    static const uint64_t V = RoundUpPow2Ull<N>::value;
public:
    static const uint64_t value = V == N ? N : V >> 1;
};

//------------------------------------------------------------------------------

//lower bound
template<unsigned N, unsigned BASE>
struct Log {
    static_assert(N > 0, "Log : N <= 0");
    static const unsigned value = N < BASE ? 0 :
                                  1 + Log<xlib::max(1u, N / BASE), BASE>::value;
};
template<unsigned BASE>
struct Log<1, BASE> {
    static const unsigned value = 0;
};

//lower bound
template<uint64_t N, uint64_t BASE>
struct LogUll {
    static_assert(N > 0, "Log : N <= 0");
    static const uint64_t value = 1 + LogUll<N / BASE, BASE>::value;
};
template<uint64_t BASE>
struct LogUll<1, BASE> {
    static const uint64_t value = 0;
};

//lower bound
template<unsigned N>
struct Log2 {
    static const unsigned value = Log<N, 2>::value;
};

template<uint64_t N>
struct Log2Ull {
    static const uint64_t value = LogUll<N, 2>::value;
};

template<unsigned N>
struct CeilLog2 {
    static const unsigned value = Log2<RoundUpPow2<N>::value>::value;
};
template<uint64_t N>
struct CeilLog2Ull {
    static const uint64_t value = Log2Ull<RoundUpPow2Ull<N>::value>::value;
};

template<unsigned N, unsigned BASE>
struct CeilLog {
private:
    static const unsigned LOG = Log<N, BASE>::value;
public:
    static const unsigned value = Pow<BASE, LOG>::value == N ? LOG : LOG + 1;
};

//------------------------------------------------------------------------------

template<unsigned LOW, unsigned HIGH>
struct ProductSequence {
    static const unsigned value = LOW * ProductSequence<LOW + 1, HIGH>::value;
};
template<unsigned LOW>
struct ProductSequence<LOW, LOW> {
    static const unsigned value = LOW;
};

template<unsigned N, unsigned K>
struct BinomialCoeff {
static_assert(N >= 0 && K >= 0 && K <= N, "BinomialCoeff");
private:
    static const unsigned MIN = xlib::min(K, N - K);
    static const unsigned MAX = xlib::max(K, N - K);
public:
    static const unsigned value = ProductSequence<MAX + 1, N>::value /
                                  xlib::factorial(MIN);
};
template<unsigned N>
struct BinomialCoeff<N ,N> {
    static const unsigned value = 1;
};

template<unsigned N, unsigned HIGH>
struct GeometricSerie {
    static const unsigned value = (Pow<N, HIGH + 1>::value - 1) / (N - 1);
};

} // namespace xlib
