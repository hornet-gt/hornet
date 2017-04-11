/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
namespace xlib {

template<unsigned A, unsigned B, unsigned... ARGS>
struct Max {
    static const unsigned value = Max<A, Max<B, ARGS...>::value>::value;
};
template<unsigned A, unsigned B>
struct Max<A, B> {
    static const unsigned value = A > B ? A : B;
};

template<unsigned A, unsigned B, unsigned... ARGS>
struct Min {
    static const unsigned value = Min<A, Min<B, ARGS...>::value>::value;
};
template<unsigned A, unsigned B>
struct Min<A, B> {
    static const unsigned value = A < B ? A : B;
};

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

template<uint64_t N>
struct IsPower2 {
    static const bool value = (N != 0) && !static_cast<bool>( (N & (N - 1)) );
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
                                  1 + Log<Max<1, N / BASE>::value, BASE>::value;
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

template<unsigned N>
struct Factorial {
    static_assert(N >= 0, "Factorial : N < 0");
    static const unsigned value = N * Factorial<N - 1>::value;
};
template<>
struct Factorial<0> {
    static const unsigned value = 1;
};

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
    static const unsigned MIN = Min<K, N - K>::value;
    static const unsigned MAX = Max<K, N - K>::value;
public:
    static const unsigned value = ProductSequence<MAX + 1, N>::value /
                                  Factorial<MIN>::value;
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
