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

 #if defined(__NVCC__)
     #define ALIGN(bytes) __align__(bytes)
     #define HOST_DEVICE  __host__ __device__ __forceinline__

     #define CONST_EXPR
     #define CONST_EXPR_ASSERT(a) assert(a)
 #else
     #define ALIGN(bytes) alignas(bytes)
     #define HOST_DEVICE inline

     #define CONST_EXPR // constexpr
     #define ENABLE_CONST_EXPR
     #define CONST_EXPR_ASSERT(a)
 #endif

 #if defined(__NVCC__) || defined(__GNUG__) || defined(__CLANG__)
     #define RESTRICT __restrict__
 #elif defined(_MSC_VER)
     #define RESTRICT __restrict
 #else
     #define RESTRICT
     #pragma message("RESTRICT not defined")
 #endif
