#!/bin/bash
for graph in "$2"*.bin; do
    echo $graph
    "$1" $graph $@
    #| grep "avg. elapsed"
done
