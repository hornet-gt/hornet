cmake_minimum_required(VERSION 3.2)
project( custinger )

set(CUDA_HOST_COMPILER /usr/bin/gcc-4.8)  
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(include/alg/static_triangle_counting)

# Adding external libraries
INCLUDE_DIRECTORIES(externals/stx-btree/include)
INCLUDE_DIRECTORIES(externals/moderngpu/src/moderngpu)
INCLUDE_DIRECTORIES(externals/cub/cub/)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# set(CMAKE_BUILD_TYPE Debug)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

LIST(APPEND CUDA_NCCC_FLAGS --compiler-options)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS "-O3;-DVERBOSE;")
LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -Wall -D_FORCE_INLINES; --expt-extended-lambda;-use_fast_math")

set(mgpu_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/moderngpu/include
  CACHE PATH
  "Directory to the Modern GPU include files")

set(mgpu_SOURCE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/moderngpu/src
  CACHE PATH
  "Directory to the Modern GPU source files")



cuda_add_library(custinger
          src/cuStinger.cpp         
          src/cuStinger.cu
          src/update.cpp
          src/update.cu
          src/cuStingerInsertions.cu
          src/cuStingerInsertions.cpp
          src/cuStingerDeletions.cu
          src/memoryManager.cpp
          src/alloc.cu
          src/graphreader.cpp
          src/timer.c
        )

cuda_add_executable(cuMain 
          src/main.cpp
           )


TARGET_LINK_LIBRARIES(cuMain custinger  /usr/local/cuda/lib64/libcudart.so)
