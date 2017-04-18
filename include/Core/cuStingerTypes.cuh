///@file

#pragma once

namespace cu_stinger {

class Edge;
class VertexSet;
class VertexIt;
class EdgeIt;

//==============================================================================

class Vertex {
friend class VertexSet;
public:
    /**
     * @brief
     */
    __device__ __forceinline__
    Vertex(id_t index) noexcept;

    /**
     * @brief
     * @return
     */
    __device__ __forceinline__
    degree_t degree() const noexcept;

    /**
     * @fn T field() const noexcept
     * @brief
     * @details **example:**
     * @code{.cpp}
     *      Vertex vertex = ...
     *      auto vertex_label = vertex.field<0>();
     * @endcode
     */
    template<int INDEX>
    __device__ __forceinline__
    typename std::tuple_element<INDEX, VertexTypes>::type
    field() const noexcept;

    /**
     * @brief
     * @return
     */
    __device__ __forceinline__
    Edge edge(off_t index) const noexcept;
private:
    byte_t* _ptrs[NUM_VTYPES];
};

//==============================================================================

class Edge {
    friend class Vertex;
public:
    /**
     * @brief
     * @return
     */
    __device__ __forceinline__
    id_t dst() const noexcept;

    using EnableWeight = typename std::conditional<(NUM_ETYPES > 1),
                                                    int, void>::type;
    /**
     * @brief
     * @details **example:**
     * @code{.cpp}
     *      Edge edge = ...
     *      auto edge_weight = edge.weight();
     * @endcode
     */
    template<typename T = EnableWeight>
    __device__ __forceinline__
    typename std::tuple_element<(NUM_ETYPES > 1 ? 1 : 0), edge_t>::type
    weight() const noexcept;

    /**
     * @brief
     * @return
     */
    __device__ __forceinline__
    typename std::tuple_element<(NUM_VTYPES > 2 ? 2 : 0), vertex_t>::type
    time_stamp1() const noexcept;

    /**
     * @brief
     * @return
     */
    __device__ __forceinline__
    typename std::tuple_element<(NUM_VTYPES > 3 ? 3 : 0), vertex_t>::type
    time_stamp2() const noexcept;

    /**
     * @brief
     * @details **example:**
     * @code{.cpp}
     *      Edge edge = ...
     *      auto edge_label = edge.field<0>();
     * @endcode
     */
    template<int INDEX>
    __device__ __forceinline__
    typename std::tuple_element<INDEX, EdgeTypes>::type
    field() const noexcept;

private:
    byte_t* _ptrs[NUM_ETYPES];

    __device__ __forceinline__
    Edge(byte_t* block_ptr, off_t index) noexcept;
};

//==============================================================================

class VertexSet {
public:
    __device__ __forceinline__
    Vertex operator[](id_t index) const noexcept;

    __device__ __forceinline__
    VertexIt begin() const noexcept;

    __device__ __forceinline__
    VertexIt end() const noexcept;
};

//==============================================================================

class VertexIt {

};

//==============================================================================

class EdgeIt {

};

} // namespace cu_stinger

#include "cuStingerTypes.i.cuh"
