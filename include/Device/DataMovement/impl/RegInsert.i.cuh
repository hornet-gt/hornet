/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

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
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
namespace xlib {

template<>
template<typename T>
__device__ __forceinline__
void Reg<RegMode::SIMPLE>::insert(T* queue, int& size, T item) {
    queue[size++] = item;
}

template<>
template<typename T>
__device__ __forceinline__
void Reg<RegMode::UNROLL>::insert(T* queue, int& size, T item) {
    switch (size) {
        case 0: queue[0] = item; break;
        case 1: queue[1] = item; break;
        case 2: queue[2] = item; break;
        case 3: queue[3] = item; break;
        case 4: queue[4] = item; break;
        case 5: queue[5] = item; break;
        case 6: queue[6] = item; break;
        case 7: queue[7] = item; break;
        case 8: queue[8] = item; break;
        case 9: queue[9] = item; break;
        case 10: queue[10] = item; break;
        case 11: queue[11] = item; break;
        case 12: queue[12] = item; break;
        case 13: queue[13] = item; break;
        case 14: queue[14] = item; break;
        case 15: queue[15] = item; break;
        case 16: queue[16] = item; break;
        case 17: queue[17] = item; break;
        case 18: queue[18] = item; break;
        case 19: queue[19] = item; break;
        case 20: queue[20] = item; break;
        case 21: queue[21] = item; break;
        case 22: queue[22] = item; break;
        case 23: queue[23] = item; break;
        case 24: queue[24] = item; break;
        case 25: queue[25] = item; break;
        case 26: queue[26] = item; break;
        case 27: queue[27] = item; break;
        case 28: queue[28] = item; break;
        case 29: queue[29] = item; break;
        case 30: queue[30] = item; break;
        case 31: queue[31] = item; break;
        case 32: queue[32] = item; break;
        case 33: queue[33] = item; break;
        case 34: queue[34] = item; break;
        case 35: queue[35] = item; break;
        case 36: queue[36] = item; break;
        case 37: queue[37] = item; break;
        case 38: queue[38] = item; break;
        case 39: queue[39] = item; break;
        case 40: queue[40] = item; break;
        case 41: queue[41] = item; break;
        case 42: queue[42] = item; break;
        case 43: queue[43] = item; break;
        case 44: queue[44] = item; break;
        case 45: queue[45] = item; break;
        case 46: queue[46] = item; break;
        case 47: queue[47] = item; break;
        case 48: queue[48] = item; break;
        case 49: queue[49] = item; break;
        case 50: queue[50] = item; break;
        case 51: queue[51] = item; break;
        case 52: queue[52] = item; break;
        case 53: queue[53] = item; break;
        case 54: queue[54] = item; break;
        case 55: queue[55] = item; break;
        case 56: queue[56] = item; break;
        case 57: queue[57] = item; break;
        case 58: queue[58] = item; break;
        case 59: queue[59] = item; break;
        case 60: queue[60] = item; break;
        case 61: queue[61] = item; break;
        case 62: queue[62] = item; break;
        case 63: queue[63] = item; break;
    }
    size++;
}

} // namespace xlib
