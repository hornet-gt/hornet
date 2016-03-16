#pragma once




#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

typedef int32_t* int32_tPtr;
typedef int32_tPtr* int32_tPtrPtr;



void update(cuStinger &custing, BatchUpdate &bu);

void reAllocateMemoryAfterSweep1(cuStinger &custing, BatchUpdate &bu);


