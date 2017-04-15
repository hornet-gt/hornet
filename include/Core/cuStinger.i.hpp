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
#include "cuStinger.hpp"

namespace cu_stinger {

template<unsigned INDEX, typename T, typename... TArgs>
void cuStinger::insertVertexData(T* vertex_data, TArgs... args) noexcept {
    static_assert(INDEX != 0 || sizeof...(TArgs) + 2 == NUM_VERTEX_TYPES,
                  "Number of Vertex data type not correct");
    using R = typename std::tuple_element<INDEX, VertexTypes>;
    static_assert(std::is_same<T, R>::value, "Incorrect Vertex data type");

    _vertex_data[INDEX] = vertex_data;
    insertVertexData<INDEX + 1>(args...);
}

template<unsigned INDEX>
void cuStinger::insertVertexData() noexcept {}

template<unsigned INDEX, typename T, typename... TArgs>
void cuStinger::insertEdgeData(T* edge_data, TArgs... args) noexcept {
    static_assert(INDEX != 0 || sizeof...(TArgs) + 2 == NUM_EDGE_TYPES,
                  "Number of Edge data type not correct");
    using R = typename std::tuple_element<INDEX, VertexTypes>;
    static_assert(std::is_same<T, R>::value, "Incorrect Edge data type");

    _edge_data[INDEX] = edge_data;
    insertVertexData<INDEX + 1>(args...);
}

template<unsigned INDEX>
void cuStinger::insertEdgeData() noexcept {}

} // namespace cu_stinger
