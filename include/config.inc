/**
 * @brief Example of cuStinger configuration
 * @file
 */
/**
 * @brief vertex id type
 * @remark `id_t` must be *integral* type
 */
using vid_t = int;

/**
 * @brief offset type
 * @remark `offset_t` must be *integral* type
 */
using eoff_t = int;

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
