/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */

#include "Util/BatchFunctions.hpp"
#include "Support/Numeric.hpp"
#include <utility>
#include <chrono>
#include <random>

void generateInsertBatch(length_t* batch_src, length_t* batch_dest,
                         int batch_size, const graph::GraphStd<>& graph,
                         BatchProperty prop) {

    if (!prop.weighted) {
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<length_t> distribution(0, graph.nV() - 1);
        for (int i = 0; i < batch_size; i++) {
            batch_src[i]  = distribution(gen);
            batch_dest[i] = distribution(gen);
        }
    }
    else {
        xlib::WeightedRandomGenerator<length_t>
            weighted_gen(graph.out_degrees_array(), graph.nV());
        for (int i = 0; i < batch_size; i++) {
            batch_src[i]  = weighted_gen.get();
            batch_dest[i] = weighted_gen.get();
        }
    }

    if (prop.print || prop.sort) {
        auto tmp_batch = new std::pair<id_t, id_t>[batch_size];
        for (int i = 0; i < batch_size; i++)
            tmp_batch[i] = std::make_pair(batch_src[i], batch_dest[i]);

        std::sort(tmp_batch, tmp_batch + batch_size);
        if (prop.sort) {
            for (int i = 0; i < batch_size; i++) {
                batch_src[i]  = tmp_batch[i].first;
                batch_dest[i] = tmp_batch[i].second;
            }
        }
        if (prop.print) {
            std::cout << "Batch:\n";
            for (int i = 0; i < batch_size; i++) {
                std::cout << "(" << tmp_batch[i].first << ","
                          << tmp_batch[i].second << ")\n";
            }
            std::cout << std::endl;
        }
        delete[] tmp_batch;
    }
}


#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>

void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D, dxor128_env_t * env);

/// Generate an edge list of batch updates using the RMAT graph random edge generator.
void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D, dxor128_env_t * env){
    int64_t src,dst;
    int scale = (int)log2(double(nv));
    for(int32_t e=0; e<numEdges; e++){
        rmat_edge(&src,&dst,scale, A,B,C,D,env);
        edgeSrc[e] = src;
        edgeDst[e] = dst;
    }
}

double dxor128(dxor128_env_t * e) {
  unsigned t=e->x^(e->x<<11);
  e->x=e->y; e->y=e->z; e->z=e->w; e->w=(e->w^(e->w>>19))^(t^(t>>8));
  return e->w*(1.0/4294967296.0);
}

void dxor128_init(dxor128_env_t * e) {
  e->x=123456789;
  e->y=362436069;
  e->z=521288629;
  e->w=88675123;
}

void dxor128_seed(dxor128_env_t * e, unsigned seed) {
  e->x=123456789;
  e->y=362436069;
  e->z=521288629;
  e->w=seed;
}


void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D, dxor128_env_t * env)
{
  int64_t i = 0, j = 0;
  int64_t bit = ((int64_t) 1) << (SCALE - 1);

  while (1) {
    const double r =  ((double) rand() / (RAND_MAX));//dxor128(env);
    if (r > A) {                /* outside quadrant 1 */
      if (r <= A + B)           /* in quadrant 2 */
        j |= bit;
      else if (r <= A + B + C)  /* in quadrant 3 */
        i |= bit;
      else {                    /* in quadrant 4 */
        j |= bit;
        i |= bit;
      }
    }
    if (1 == bit)
      break;

    /*
      Assuming R is in (0, 1), 0.95 + 0.1 * R is in (0.95, 1.05).
      So the new probabilities are *not* the old +/- 10% but
      instead the old +/- 5%.
    */
    A *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    B *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    C *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    D *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    /* Used 5 random numbers. */

    {
      const double norm = 1.0 / (A + B + C + D);
      A *= norm;
      B *= norm;
      C *= norm;
    }
    /* So long as +/- are monotonic, ensure a+b+c+d <= 1.0 */
    D = 1.0 - (A + B + C);

    bit >>= 1;
  }
  /* Iterates SCALE times. */
  *iout = i;
  *jout = j;
}
