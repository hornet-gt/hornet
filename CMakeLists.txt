cmake_minimum_required(VERSION 2.8)
project( custinger )

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(include)

#message("message here")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

#SET(CUDA_HOST_COMPILATION_CPP ON)
#SET(CUDA_SEPERABLE_COMPILATION ON)

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -fPIC")

cuda_add_executable(custinger 
					src/main.cpp
					src/alloc.cu
					src/graphreader.cpp
					src/update.cu
				   )

TARGET_LINK_LIBRARIES(custinger /usr/local/cuda/lib64/libcudart.so)

#LIST(APPEND CMAKE_CXX_FLAGS "std=c++0x -O3 -ffastmath -Wall")

LIST(APPEND CUDA_NCCC_FLAGS --compiler-options -fno-strict-aliasing -use-fast-math)
LIST(APPEND CUDA NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA NVCC_FLAGS -gencode arch=compute_21,code=sm_21)
LIST(APPEND CUDA NVCC_FLAGS "-std=c++11;-03;-DVERBOSE")

#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else()
#        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()
#
