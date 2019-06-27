#ifndef EDGE_OPERATIONS_CUH
#define EDGE_OPERATIONS_CUH
enum class EdgeUpdateOperation {
    ignore,
    update,//Overwrite meta-data of matching edges in graph with that of batch
    sum //Sum meta-data of matching edges in graph with that of batch
};
#endif
