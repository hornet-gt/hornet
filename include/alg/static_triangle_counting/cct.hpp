#ifndef _CU_STATIC_CCT_INCLUDE_HPP_
#define _CU_STATIC_CCT_INCLUDE_HPP_


#include <stdio.h>
#include <inttypes.h>

#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

typedef int32_t triangle_t;


void callDeviceAllTriangles(cuStinger& custing,
    triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
    const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);





#endif
