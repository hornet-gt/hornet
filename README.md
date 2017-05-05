# cuStingerAlg #

The repository provides the graph algorithms implemented over the cuStinger
data structure.

For additional information please refer to [**cuStinger repository**]( https://github.com/FedericoUnivr/cuStinger).

## Getting Started ##

The document is organized as follows:

* [Quick start](#quick-start)
* [cuStinger Algorithms](#custinger-algorithms)
* [Performance](#performance)
* [cuStinger Algorithms Code Statistics](#custinger-algorithms-code-statistics)
* [Acknowledgements](#acknowledgements)
* [License](#licence)

### Quick Start ###

The following basic steps are required to build and execute the cuStinger algorithms:
```bash
git clone --recursive https://github.com/FedericoUnivr/cuStingerAlg.git
cd cuStingerAlg/build
cmake ..
make -j
```

### cuStinger Algorithms ###

|           Algorithm                 |    Static     | Dynamic  |
| :-----------------------------------|:-------------:|:--------:|
| (BFS) Breadth-first Search          |     yes       | on-going |
| (SSSP) Single-Source Shortest Path  |     yes       | on-going |
| (CC) Connected Components           |     yes       | on-going |
| (MST) Minimum Spanning Tree         |    to-do      |  to-do   |
| (BC) Betweenness Centrality         |     yes       | on-going |
| (PG) Page Rank                      |     yes       | on-going |
| (TC) Triangle Counting              |     yes       |    yes   |
| (KC) Katz Centrality                |     yes       |    yes   |
| (KC) james                          |  on-going     |    to-do |


## Performance ##

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

|           Algorithm                 | CPU<sup>3</sup> | GPU <sup>4</sup> |  Speedup |
| :-----------------------------------|:---------------:|:----------------:|:--------:|
| (BFS) Breadth-first Search          |                 |                  |          |
| (SSSP) Single-Source Shortest Path  |                 |                  |          |
| (CC) Connected Components           |                 |                  |          |
| (MST) Minimum Spanning Tree         |                 |                  |          |
| (BC) Betweenness Centrality         |                 |                  |          |
| (PG) Page Rank                      |                 |                  |          |
| (TC) Triangle Counting              |                 |                  |          |
| (KC) Katz Centrality                |                 |                  |          |

<sup>3</sup> Intel ...   <br>
<sup>4</sup> NVidia Tesla K80 ..

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


### cuStinger Algorithms Code Statistics ###

|      Static Algorithm               | lines of code |
| :-----------------------------------|:-------------:|
| (BFS) Breadth-first Search          |     0       |
| (SSSP) Single-Source Shortest Path  |     0       |
| (CC) Connected Components           |     0       |
| (MST) Minimum Spanning Tree         |     0       |
| (BC) Betweenness Centrality         |     0       |
| (PG) Page Rank                      |     0       |
| (TC) Triangle Counting              |     0       |
| (KC) Katz Centrality                |     0       |

|      Dynamic Algorithm              | lines of code |
| :-----------------------------------|:-------------:|
| (BFS) Breadth-first Search          |     0       |
| (SSSP) Single-Source Shortest Path  |     0       |
| (CC) Connected Components           |     0       |
| (MST) Minimum Spanning Tree         |     0       |
| (BC) Betweenness Centrality         |     0       |
| (PG) Page Rank                      |     0       |
| (TC) Triangle Counting              |     0       |
| (KC) Katz Centrality                |     0       |


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
