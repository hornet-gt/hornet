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
void cuStinger::insertVertexData(const T* vertex_data, TArgs... args) noexcept {
    using R = typename std::tuple_element<INDEX, VertexTypes>::type;

    static_assert(INDEX != 0 || sizeof...(TArgs) + 1 == NUM_EXTRA_VTYPES,
                  "Number of Vertex data type not correct");
    static_assert(std::is_same<typename std::remove_cv<T>::type,
                               typename std::remove_cv<R>::type>::value,
                  "Incorrect Vertex data type");

    _vertex_data_ptr[INDEX] = const_cast<byte_t*>(
                                 reinterpret_cast<const byte_t*>(vertex_data));
    insertVertexData<INDEX + 1>(args...);
}

template<unsigned INDEX>
void cuStinger::insertVertexData() noexcept { _vertex_init = true; }

template<unsigned INDEX, typename T, typename... TArgs>
void cuStinger::insertEdgeData(const T* edge_data, TArgs... args) noexcept {
    using R = typename std::tuple_element<INDEX, EdgeTypes>::type;

    static_assert(INDEX != 0 || sizeof...(TArgs) + 1 == NUM_EXTRA_ETYPES,
                  "Number of Edge data type not correct");
    static_assert(std::is_same<typename std::remove_cv<T>::type,
                               typename std::remove_cv<R>::type>::value,
                  "Incorrect Edge data type");

    _edge_data_ptr[INDEX] = const_cast<byte_t*>(
                                 reinterpret_cast<const byte_t*>(edge_data));
    insertEdgeData<INDEX + 1>(args...);
}

template<unsigned INDEX>
void cuStinger::insertEdgeData() noexcept { _edge_init = true; }

} // namespace cu_stinger
