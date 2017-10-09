# HornetAlg #

The repository provides the graph algorithms implemented on top Hornet
data structure.

For additional information concerning the data structure and its APIs please refer to [**Hornet repository**](https://github.com/hornet-gt/hornet).

## Getting Started ##

The document is organized as follows:

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [Hornet Algorithms](#hornet-algorithms)
* [Performance](#performance)
* [Hornet Algorithms Lines of Code](#hornet-algorithms-lines-of-code)
* [Code Documentation](#code-documentation)
* [Reporting bugs and contributing](#reporting-bugs-and-contributing)
* [Publications](#publications)
* [Hornet Developers](#hornet-developers)
* [Acknowledgements](#acknowledgements)
* [License](#licence)

### Requirements ###

* [Nvidia Modern GPU](https://developer.nvidia.com/cuda-gpus) (compute capability &ge; 3.0): Kerpler, Maxwell, Pascal, Volta architectures.
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) 9 or greater.
* GCC or [Clang](https://clang.llvm.org) host compiler with support for C++14.
  Note: the compiler must be compatible with the related CUDA toolkit version.
  For more information see [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
* [CMake](https://cmake.org) v3.6 or greater.
* 64-bit Operating System (Ubuntu 16.04 or above suggested).

### Quick Start ###

The following basic steps are required to build and execute the Hornet algorithms:
```bash
git clone --recursive https://github.com/hornet-gt/hornetsnest
cd hornetsnest/build
cmake ..
make -j
```

By default the cuda compiler `nvcc` uses `gcc/g++` found in the current
execution search path (`cc --version` to get the default compiler).
To force a different host compiler to compile plain C++ files (`*.cpp`) set the
following environment variables:
 ```bash
CC=<path_to_host_C_compiler>
CXX=<path_to_host_C++_compiler>
```

To force a different host compiler to compile host side `nvcc` code (`*.cu`)
substitute `cmake ..` with
 ```bash
cmake -DCUDAHC=<path_to_host_C++_compiler>
```
Note: host compiler and host side `nvcc` compiler may be different.
The host side `nvcc` compiler must be compatible with the current CUDA toolkit
version installed on the system
(see [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)).

## Hornet Algorithms ##

|           Algorithm                 |    Static     | Dynamic  |
| :-----------------------------------|:-------------:|:--------:|
| (BFS) Breadth-first Search          |     yes       | on-going |
| (SSSP) Single-Source Shortest Path  |     yes       | on-going |
| (CC) Connected Components           |     yes       | on-going |
| (SCC) Strongly Connected Components |    to-do      |  to-do   |
| (MST) Minimum Spanning Tree         |    to-do      |  to-do   |
| (BC) Betweeness Centrality          |   on-going    | on-going |
| (PG) Page Rank                      |     yes       | on-going |
| (TC) Triangle Counting              |     yes       |   yes    |
| (KC) Katz Centrality                |     yes       |   yes    |
| (MIS) Maximal Independent Set       |   on-going    |  to-do   |
| (MF) Maximum Flow                   |    to-do      |  to-do   |
| (CC) Clustering Coeffient           |    to-do      |  to-do   |
| (ST) St-Connectivity                |    to-do      |  to-do   |
| (TC) Transitive Closure             |    to-do      |  to-do   |
| Community Detection                 |    to-do      |  to-do   |
| Temporal Motif Finding              |   on-going    |  to-do   |
| Sparse Vector-Matrix Multiplication |     yes       |  to-do   |
| Jaccard indices                     |    to-do      |  to-do   |
| Energy/Parity Game                  |   on-going    |  to-do   |

## Performance ##

##### CPU vs. GPU #####

|           Algorithm                 | CPU<sup>1</sup> | GPU <sup>2</sup> |  Speedup |
| :-----------------------------------|:---------------:|:----------------:|:--------:|
| (BFS) Breadth-first Search          |                 |                  |          |
| (SSSP) Single-Source Shortest Path  |                 |                  |          |
| (CC) Connected Components           |                 |                  |          |
| (MST) Minimum Spanning Tree         |                 |                  |          |
| (BC) Betweenness Centrality         |                 |                  |          |
| (PG) Page Rank                      |                 |                  |          |
| (TC) Triangle Counting              |                 |                  |          |
| (KC) Katz Centrality                |                 |                  |          |

<sup>1</sup> Intel ...   <br>
<sup>2</sup> NVidia Tesla P100 ..

##### Static vs. Dynamic #####

|           Algorithm                 |     Static      |      Dynamic     |  Speedup |
| :-----------------------------------|:---------------:|:----------------:|:--------:|
| (BFS) Breadth-first Search          |                 |                  |          |
| (SSSP) Single-Source Shortest Path  |                 |                  |          |
| (CC) Connected Components           |                 |                  |          |
| (MST) Minimum Spanning Tree         |                 |                  |          |
| (BC) Betweenness Centrality         |                 |                  |          |
| (PG) Page Rank                      |                 |                  |          |
| (TC) Triangle Counting              |                 |                  |          |
| (KC) Katz Centrality                |                 |                  |          |

## Hornet Algorithms Lines of Code ##

|         Algorithm                   | Static (A) | Static (B) | Dynamic (A) |
| :-----------------------------------|:----------:|:----------:|:-----------:|
| (BFS) Breadth-first Search          |     4      |     6      |             |
| (SSSP) Single-Source Shortest Path  |            |            |             |
| (CC) Connected Components           |            |            |             |
| (MST) Minimum Spanning Tree         |            |            |             |
| (BC) Betweenness Centrality         |            |            |             |
| (PG) Page Rank                      |            |            |             |
| (TC) Triangle Counting              |            |            |             |
| (KC) Katz Centrality                |            |            |             |

<sup>(A) lines of code required for the algorithm </sup><br>
<sup>(B) lines of code required for the operators </sup>

### Code Documentation ###

The code documentation is located in the `docs` directory of Hornet data
structure directory (*doxygen* html format).
The documentation is also accessible online [**here**.](https://federicounivr.github.io/Hornet/)

### Reporting bugs and contributing ###

If you find any bugs please report them by using the repository (github **issues** panel).
We are also ready to engage in improving and extending the framework if you request new features.

## Publications ##

* Oded Green, David A. Bader, **"cuSTINGER: Supporting dynamic graph algorithms
  for GPUs"**,
  IEEE High Performance Extreme Computing Conference (HPEC), 13-15 September,
  2016, Waltham, MA, USA, pp. 1-6.
  [link](https://www.researchgate.net/publication/308174457_cuSTINGER_Supporting_dynamic_graph_algorithms_for_GPUs)
* Oded Green, James Fox, Euna Kim, Federico Busato, Nicola Bombieri,
  Kartik Lakhotia, Shijie Zhou, Shreyas Singapura, Hanqing Zeng,
  Rajgopal Kannan, Viktor Prasanna, David A. Bader,
  **"Quickly Finding a Truss in a Haystack"**,
  IEEE/Amazon/DARPA Graph Challenge, \**Innovation Awards*\*.
* Devavret Makkar, David A. Bader, Oded Green,
  **Exact and Parallel Triangle Counting in Streaming Graphs**,
  IEEE Conference on High Performance Computing, Data, and Analytics (HiPC),
  18-21 December 2017, Jaipur, India, pp. 1-10.

---
### <center>If you find this software useful in academic work, please acknowledge Hornet. </center> ###
***

## Hornet Developers ##

##### Data Structure ######

* `Federico Busato`, Ph.D. Student, University of Verona (Italy)
* `Oded Green`, Researcher, Georgia Institute of Technology

##### Algorithms ######

* `Federico Busato`, Ph.D. Student, University of Verona (Italy)
* `Oded Green`, Researcher, Georgia Institute of Technology
* `James Fox`, Ph.D. Student, Georgia Institute of Technology : *Maximal Independent Set*, *Temporal Motif Finding*
* `Devavret Makkar`, Ph.D. Student, Georgia Institute of Technology : *Triangle Counting*
* `Elisabetta Bergamini`, Ph.D. Student, Karlsruhe Institute of Technology (Germany) : *Katz Centrality*
* `Euna Kim`, Ph.D. Student, Georgia Institute of Technology : *Dynamic PageRank*
* ...

## Acknowledgements ##

* Grant...

## License ##

> BSD 3-Clause License
>
> Copyright (c) 2017, Hornet
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>   contributors may be used to endorse or promote products derived from
>   this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
