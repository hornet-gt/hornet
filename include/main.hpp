// #pragma once

#ifndef _CUSTINGER_INCLUDE_H
#define _CUSTINGER_INCLUDE_H

#include <iostream>
#include <string>

using namespace std;

#include "utils.hpp"
#include "cuStinger.hpp"
#include "update.hpp"

typedef int32_t* int32_tPtr;
typedef int32_tPtr* int32_tPtrPtr;


void update(cuStinger &custing, BatchUpdate &bu);

void reAllocateMemoryAfterSweep1(cuStinger &custing, BatchUpdate &bu);

#endif

