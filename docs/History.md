*Version v2 - Federico Busato*

* GraphIO
    - Read 7 different formats:
        - Matrix Market
        - Metis (Dimacs10th)
        - SNAP
        - Dimacs9th
        - Binary
        - Koblenz Network Collection
        - Network Data Repository
    - Write 2 formats:
        - Matrix Market
        - Binary
    - Automatically detect directed/undirected
* Updaded README.md
* Updaded CMakeLists.txt
    - Automatically detect the Compute Capability and number of SMs
    - Automatically dectect maximum number of threads
    - Optimized "Release" compiler flags
    - Fast switch between compile types (debug, info, release)
* Updaded project directories structure
* Added empty build direcory
* Added doxygen configuration file (user/developer)
* Added script to download University of Florida sparse matrices
* Added a three small graph examples (.mtx, .gr)
* Added online documentation

*April, 2017: Version v2 - Federico Busato*

## Tasks done ##

* print function for small graph
* batch generation: uniform/weighted sorted/unsorted
* minimal external configuration "config.inc"
* improved load balancing: Single pass and 3-steps
* new memory management
* graph consistency checking
* store cuStinger graph snapshot to disk
* CSR representation in addition to cuStinger
* Parsing of weighted graphs
* Two-level and Multi-level Parallel Queues

*May, 2017: Version v2 - Federico Busato*

## Tasks done ##

* Graph Transpose almost completed
* cuStinger ID

## Tasks to do ##
* Code Related:
    - Consistent APIs
    - Working algorithms
    - Compression
    - Insert/delete new vertices
    - Batch equal operators
    - Slightly improve load balancing: upper_bound early exit, variable host partitioning
    - Kernel Range-loop iterators
    - Understand performance difference between CSR and Helix
    - SpMV
    - Faster Initialization (device allocation)
    - Support 2^64 edges
    - Support Incoming Edges
    - GPU Graph Transpose
    - GPU Graph Generation
    - CPU Support
* Others
    - Regression test (ctest)
    - Add class diagram to doxygen

 ## NOTES ##

* Generic Queue vs. Custom Queue:
    - Generic Queue requires 4x times memory movements in global memory
    - Generic Queue requires a external library to compute the workload prefix-sum

* Pass function vs. struct as Operator (or Lambda):
    - Function must have fixed types. It requires explicit type casting
    - void* optional_data requires an additional global memory access
    - Operator header must knows Vertex, Edge, vid_t, eoff_t types
    - All libraries (std, thrust, cub, moderngpu, etc.) use Struct operator
    (Function may be slower since __forceinline__ may be lost in template)

* Extended Lambda expression limitations:
    - Cannot be used in class inline costructur
    - It captures "this" by default, also with [=] capture list
    (May be slower than struct static function since __forceinline__ and
     __restrict__ may be lost in template)
