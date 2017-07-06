/**
 * @brief High-level API to access to cuStinger data (Vertex, Edge)
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
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

#include "Core/cuStingerDeviceData.cuh" //cuStingerDevice
#include "Core/RawTypes.hpp"

namespace custinger {

class Edge;

//==============================================================================

class Vertex {
    friend class VertexSet;
    using WeightT = typename std::tuple_element<(NUM_ETYPES > 1 ? 1 : 0),
                                                 edge_t>::type;
public:
    /**
     * @internal
     * @brief Default costructor
     * @param[in] data cuStinger device data
     */
    __device__ __forceinline__
    Vertex(cuStingerDevice data, vid_t index);

    /**
     * @brief id of the vertex
     * @return id of the vertex
     */
    __device__ __forceinline__
    vid_t id() const;

    /**
     * @brief degree of the vertex
     * @return degree of the vertex
     */
    __device__ __forceinline__
    degree_t degree() const;

    /**
     * @brief  value of a user-defined vertex field
     * @tparam INDEX index of the user-defined vertex field to return
     * @return value of the user-defined vertex field at the index `INDEX`
     *         (type at the index `INDEX` in the `EdgeTypes` list)
     * @remark the method does not compile if the `VertexTypes` list does not
     *         contain atleast `INDEX` fields
     * @details **Example:**
     * @code{.cpp}
     *      auto vertex_label = vertex.field<0>();
     * @endcode
     */
    template<int INDEX>
    __device__ __forceinline__
    typename std::tuple_element<INDEX, VertexTypes>::type
    field() const;

    /**
     * @brief Get an edge associeted to the vertex
     * @param[in] index index of the edge
     * @return edge at index `index`
     * @warning `index` must be in the range \f$0 \le index < degree\f$.
     * The behavior is undefined otherwise.
     */
    __device__ __forceinline__
    Edge edge(degree_t index) const;

    /**
     * @internal
     * @brief return the limit (upper approximation) of the degree of the vertex
     * @return number of edges in the actual *block*
     */
    __device__ __forceinline__
    degree_t limit() const;

    /**
     * @internal
     * @brief pointer to the device degree location
     * @return pointer to the device degree location
     */
    __device__ __forceinline__
    degree_t* degree_ptr();

    /**
     * @internal
     * @brief Store an edge at a specific position in the adjacency array
     * @param[in] edge Edge to store
     * @param[in] pos Position where substite the edge
     */
    __device__ __forceinline__
    void store(const Edge& edge, degree_t pos);

    __device__ __forceinline__
    vid_t* neighbor_ptr() const;

    template<typename T = WeightT>
    __device__ __forceinline__
    WeightT* edge_weight_ptr() const;

protected:
    vid_t    _id;
    degree_t _degree;
    byte_t*  _ptrs[NUM_EXTRA_VTYPES];

private:
    VertexBasicData* _vertex_ptr;
    byte_t*          _edge_ptr;
    degree_t         _limit;
};

//==============================================================================

class Edge {
    friend class Vertex;
    using     WeightT = typename std::tuple_element<(NUM_ETYPES > 1 ? 1 : 0),
                                                     edge_t>::type;
    using TimeStamp1T = typename std::tuple_element<(NUM_ETYPES > 2 ? 2 : 0),
                                                     edge_t>::type;
    using TimeStamp2T = typename std::tuple_element<(NUM_ETYPES > 3 ? 3 : 0),
                                                     edge_t>::type;

    using     EnableWeight = typename std::conditional<(NUM_ETYPES > 1),
                                                        int, void>::type;
    using EnableTimeStamp1 = typename std::conditional<(NUM_ETYPES > 2),
                                                        int, void>::type;
    using EnableTimeStamp2 = typename std::conditional<(NUM_ETYPES > 3),
                                                        int, void>::type;

    static_assert(std::is_same<WeightT, int>::value, "T error");//???
public:
    /**
     * @brief source of the edge
     * @return source of the edge
     */
    __device__ __forceinline__
    vid_t src() const;

    /**
     * @brief destination of the edge
     * @return destination of the edge
     */
    __device__ __forceinline__
    vid_t dst() const;

    /**
     * @brief Source vertex of the edge
     * @return Source vertex
     */
    //__device__ __forceinline__
    //Vertex src() const;

    /**
     * @brief Destination vertex of the edge
     * @return Destination vertex
     */
    //__device__ __forceinline__
    //Vertex dst() const;

    /**
     * @brief weight of the edge (if it exists)
     * @return weight of the edge (first `EdgeTypes` type)
     * @remark the method is disabled if the `EdgeTypes` list does not contain
     *         atleast one field
     * @details **Example:**
     * @code{.cpp}
     *      auto edge_weight = edge.weight();
     * @endcode
     */
    template<typename T = EnableWeight>
    __device__ __forceinline__
    WeightT weight() const;

    template<typename T = EnableWeight>
    __device__ __forceinline__
    void set_weight(WeightT weight);

    /**
     * @brief first time stamp of the edge
     * @return first time stamp of the edge (second `EdgeTypes` type)
     * @remark the method is disabled if the `EdgeTypes` list does not contain
     *         atleast two fields
     */
    template<typename T = EnableTimeStamp1>
    __device__ __forceinline__
    TimeStamp1T time_stamp1() const;

    /**
     * @brief second time stamp of the edge
     * @return second time stamp of the edge (third `EdgeTypes` list type)
     * @remark the method is disabled if the `EdgeTypes` list does not contain
     *         atleast three fields
     */
    template<typename T = EnableTimeStamp2>
    __device__ __forceinline__
    TimeStamp2T time_stamp2() const;

    /**
     * @brief  value of a user-defined edge field
     * @tparam INDEX index of the user-defined edge field to return
     * @return value of the user-defined edge field at the index `INDEX`
     *         (type at the index `INDEX` in the `EdgeTypes` list)
     * @remark the method does not compile if the `EdgeTypes` list does not
     *         contain atleast `INDEX` fields
     * @details **Example:**
     * @code{.cpp}
     * Edge edge = ...
     *      auto edge_label = edge.field<0>();
     * @endcode
     */
    template<int INDEX>
    __device__ __forceinline__
    typename std::tuple_element<INDEX, EdgeTypes>::type
    field() const;

    /**
     * @internal
     * @brief Default Costrustor
     * @param[in] block_ptr pointer in the *block* to the edge
     * @param[in] index index of the edge in the adjacency array
     * @param[in] size of the *block*
     */
    __device__ __forceinline__
    Edge(byte_t* block_ptr, degree_t index, degree_t limit);

protected:
    vid_t   _dst;
    byte_t* _ptrs[NUM_EXTRA_ETYPES];
};

} // namespace custinger

#include "impl/cuStingerTypes.i.cuh"
