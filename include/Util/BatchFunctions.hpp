#pragma once

#include "GraphIO/GraphStd.hpp"
#include "Core/cuStingerDefs.hpp"

struct BatchProperty {
    bool sort, print, weighted;

    BatchProperty(bool _sort       = false,
                  bool _weighted   = false,
                  bool _print      = false) :
                        sort(_sort), weighted(_weighted), print(_print) {}
};

void generateInsertBatch(length_t* batch_src, length_t* batch_dest,
                         int batch_size, const graph::GraphStd<>& graph,
                         BatchProperty prop = BatchProperty());

//==============================================================================
typedef struct dxor128_env {
  unsigned x,y,z,w;
} dxor128_env_t;


void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D, dxor128_env_t * env);
