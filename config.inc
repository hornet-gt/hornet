/**
 * @brief Example of cuStinger configuration
 * @file
 */

/**
 * @brief vertex id type
 * @remark `id_t` must be *integral* type
 */
using id_t = int;

/**
 * @brief offset type
 * @remark `off_t` must be *integral* type
 */
using off_t = int;

/**
 * @brief list of types for additional vertex data
 * @details **Example**
 * @code{.cpp}
 *       using VertexTypes = TypeList<char, float>;
 * @endcode
 */
using VertexTypes = TypeList<>;
//using VertexTypes = TypeList<unsigned char>;

/**
 * @brief list of types for additional edge data
 * @details **Example**
 * @code{.cpp}
 *       using EdgeTypes = TypeList<uint64_t, float>;
 * @endcode
 */
using EdgeTypes = TypeList<>;
//using EdgeTypes = TypeList<uint64_t, float>;

/**
 * @brief minimum number of edges for a **block**
 * @remark `MIN_EDGES_PER_BLOCK` must be a power of two
 */
const size_t  MIN_EDGES_PER_BLOCK = 2;

/**
 * @brief number of edges for a **BlockArray**
 * @remark `EDGES_PER_BLOCKARRAY` must be a power of two
 */
const size_t EDGES_PER_BLOCKARRAY = 262144;
