# cuSTINGER

The following includes instructions on how to clone and build this repo.
It is important that you clone the repo with all its dependencies using the **recursive** option:
```
git clone --recursive https://github.com/cuStinger/cuStinger.git
```

To build the repo, proceed to ```cd``` in the clone directory and proceed as follows:
```
mkdir build && cd build
cmake ..
make -j8
```

At this point cuSTINGER is ready for use. Currently, cuSTINGER has a single demo that can be executed:
```
./cuMain inputgraph graph-name
```
For example (assuming your graphs are stored in the directory **~/dimacs**:
```
./cuMain ~/dimacs/astro-ph.graph astro-ph
```

cuSTINGER can currently read the following graph types:
* DIMACS10 - http://www.cc.gatech.edu/dimacs10/
* SNAP - http://snap.stanford.edu/


Additional information on cuSTINGER can be found in the [HPEC'16 conference paper](https://www.researchgate.net/publication/308174457_cuSTINGER_Supporting_Dynamic_Graph_Algorithms_for_GPUs).
