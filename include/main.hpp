// #pragma once

#ifndef _MAIN_INCLUDE_H
#define _MAIN_INCLUDE_H

#include <iostream>
#include <string>

using namespace std;

#include "utils.hpp"
#include "cuStinger.hpp"
#include "update.hpp"


void update(cuStinger &custing, BatchUpdate &bu);

void reAllocateMemoryAfterSweep1(cuStinger &custing, BatchUpdate &bu);

#endif

