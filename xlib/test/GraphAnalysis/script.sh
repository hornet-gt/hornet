#!/bin/bash
for graph in $1/*
do
    basename $graph
    ./build/graph_analysis $graph
done
