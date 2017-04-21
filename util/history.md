
*April, 2017: Version v2 - Federico Busato*

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
* Added doxygen configuration file
* Added script to download University of Florida sparse matrices
* Added a small graph example (.mtx)
* Code Refactoring: (Google/LLVM style)
    - Substituted tab '\t' with 4 spaces
    - 80 columns max
    - NULL --> nullptr
    - Code alignment
    - int32_t -> int
    - typename -> using
    - malloc/free -> new/delete
    - reinterpret_cast, static_cast and const_cast instead of old style cast
    - _var for private variables
    - d_var for device variables and h_var for host variables
    - ...
* cuStinger :
    - print function for small graph
    - batch generation: uniform/weighted sorted/unsorted
    - external configuration "config.inc"
- Added online documentation

TO DO:
* Code Related:
    - Kernel Interface
    - Helix Load Balancing
    - Insert new vertices
    - Batch equal operators
    - Graph Weights IO
    - Kernel Range-loop iterators
* Others
    - Regression test
    - Added Syntax.txt
    - Store the graph to disk
