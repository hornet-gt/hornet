# cuStingerAlg #

The repository provides the graph algorithms implemented over the cuStinger
data structure.

For additional information please refer to [**cuStinger repository**]( https://github.com/FedericoUnivr/cuStinger).

## Getting Started ##

The document is organized as follows:

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [cuStinger Algorithms](#custinger-algorithms)
* [Performance](#performance)
* [cuStinger Algorithms Lines of Code](#custinger-algorithms-lines-of-code)
* [Reporting bugs and contributing](#reporting-bugs-and-contributing)
* [Publications](#publications)
* [cuStinger Developers](#custinger-developers)
* [Acknowledgements](#acknowledgements)
* [License](#licence)

## Requirements ##

* [Nvidia Modern GPU](https://developer.nvidia.com/cuda-gpus) (compute capability >= 3.0): Kerpler, Maxwell, Pascal architectures.
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) v7.5 or greater. (CUDA toolkit v8.0 recommended, CUDA toolkit v7.0 supported but not tested)
* GCC or [Clang](https://clang.llvm.org) host compiler with support for C++11<sup>*</sup>.
* [CMake](https://cmake.org) v3.5 or greater.
* 64-bit Operating System (Ubuntu tested).

<sup>*The next release will support C++14 (CUDA Toolkit v9). </sup>

## Quick Start ##

The following basic steps are required to build and execute the cuStinger algorithms:
```bash
git clone --recursive https://github.com/FedericoUnivr/cuStingerAlg.git
cd cuStingerAlg/build
cmake ..
make -j
```

## cuStinger Algorithms ##

|           Algorithm                 |    Static     | Dynamic  |
| :-----------------------------------|:-------------:|:--------:|
| (BFS) Breadth-first Search          |     yes       | on-going |
| (SSSP) Single-Source Shortest Path  |     yes       | on-going |
| (CC) Connected Components           |     yes       | on-going |
| (SCC) Strongly Connected Components |    to-do      |  to-do   |
| (MST) Minimum Spanning Tree         |    to-do      |  to-do   |
| (BC) Betweeness Centrality          |     yes       | on-going |
| (PG) Page Rank                      |     yes       | on-going |
| (TC) Triangle Counting              |     yes       |   yes    |
| (KC) Katz Centrality                |     yes       |   yes    |
| (MIS) Maximal Independent Set       |    to-do      |  to-do   |
| (MF) Maximum Flow                   |    to-do      |  to-do   |
| (CC) Clustering Coeffient           |    to-do      |  to-do   |
| (ST) St-Connectivity                |    to-do      |  to-do   |
| (TC) Transitive Closure             |    to-do      |  to-do   |
| Community Detection                 |    to-do      |  to-do   |
| Temporal Motif Finding              |   on-going    |  to-do   |
| Sparse Vector-Matrix Multiplication |     yes       |  to-do   |
| Jaccard indices                     |    to-do      |  to-do   |
| Energy/Parity Game                  |    to-do      |  to-do   |

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

## cuStinger Algorithms Lines of Code ##

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

### Reporting bugs and contributing ###

If you find any bugs please report them by using the repository (github **issues** panel).
We are also ready to engage in improving and extending the framework if you request some new features.


## Publications ##

* Oded Green, David A. Bader, [*"cuStinger: Supporting dynamic graph algorithms for GPUs"*](https://www.researchgate.net/publication/308174457_cuSTINGER_Supporting_dynamic_graph_algorithms_for_GPUs),
IEEE High Performance Extreme Computing Conference (HPEC), 13-15 September, 2016, Waltham, MA, USA, pp. 1-6.


---
### <center>If you find this software useful in academic work, please acknowledge cuStinger. </center> ###
***

## cuStinger Developers ##

##### Data Structure ######

* `Oded Green`, Researcher, Georgia Institute of Technology
* `Federico Busato`, Ph.D. Student, University of Verona (Italy)

##### Algorithms ######

* `Oded Green`, Researcher, Georgia Institute of Technology
* `Federico Busato`, Ph.D. Student, University of Verona (Italy)
* `James Fox`, Ph.D. Student, Georgia Institute of Technology : *Temporal Motif Finding*
* `Devavret Makkar`, Ph.D. Student, Georgia Institute of Technology : *Triangle Counting*
* `Elisabetta Bergamini`, Ph.D. Student, Karlsruhe Institute of Technology (Germany) : *Katz Centrality*
* ...

## Acknowledgements ##

* Grant...

## License ##

> BSD 3-Clause License
>
> Copyright (c) 2017, cuStinger
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
