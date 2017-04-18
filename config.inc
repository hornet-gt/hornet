///@file

///@brief vertex id type
using  id_t = int;

///@brief offset type
using off_t = int;

 ///@brief list of types for additional vertex data
//using VertexTypes = TypeList<unsigned char>;
using VertexTypes = TypeList<>;

 ///@brief list of types for additional edge data
//using   EdgeTypes = TypeList<uint64_t, float>;
using   EdgeTypes = TypeList<>;

///@brief minimum number of edges for a **block**
const size_t  MIN_EDGES_PER_BLOCK = 2;

///@brief number of edges for a **BlockArray**
const size_t EDGES_PER_BLOCKARRAY = 128;
