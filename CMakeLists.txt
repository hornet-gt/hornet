cmake_minimum_required(VERSION 2.8)
project( custinger )

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(include/alg/static_triangle_counting)


# Adding external libraries
INCLUDE_DIRECTORIES(externals/stx-btree/include)
# INCLUDE_DIRECTORIES(externals/moderngpu/include)


#message("message here")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

#SET(CUDA_HOST_COMPILATION_CPP ON)
#SET(CUDA_SEPERABLE_COMPILATION ON)

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -fPIC")
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")

set(mgpu_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/moderngpu/include
  CACHE PATH
  "Directory to the Modern GPU include files")

set(mgpu_SOURCE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/moderngpu/src
  CACHE PATH
  "Directory to the Modern GPU source files")


cuda_add_library(cuUtils
					src/alloc.cu
					src/graphreader.cpp
          src/timer.c
				)

cuda_add_library(cuUpdate
					src/update.cpp
          src/cuStingerInsertions.cu
          src/cuStingerInsertions.cpp
          src/cuStingerDeletions.cu
          src/memoryManager.cpp
				)

cuda_add_library(cuAlgCCT
          src/alg/static_triangle_counting/cct.cu
        )
cuda_add_library(custinger
          src/cuStinger.cpp         
          src/cuStinger.cu
        )

cuda_add_executable(cuMain 
					src/main.cpp
				   )

cuda_add_executable(cuMemManager 
					src/memoryManager.cpp
				   )

set_property(
   TARGET cuMemManager
   PROPERTY COMPILE_DEFINITIONS   MEM_STAND_ALONE
   )



cuda_add_executable(cctMain
          src/alg/static_triangle_counting/cct.cpp
          src/alg/static_triangle_counting/cct_csr.cu        
        )

cuda_add_executable(cctUpdate
          src/modified.cu
          src/alg/static_triangle_counting/cct_upd.cpp
          src/alg/static_triangle_counting/cct_upd.cu
        )

TARGET_LINK_LIBRARIES(cctMain custinger cuUtils cuUpdate cuAlgCCT /usr/local/cuda/lib64/libcudart.so)

TARGET_LINK_LIBRARIES(cuMain custinger cuUtils cuUpdate cuAlgCCT /usr/local/cuda/lib64/libcudart.so)

TARGET_LINK_LIBRARIES(cuMemManager custinger cuUtils cuUpdate cuAlgCCT /usr/local/cuda/lib64/libcudart.so)

TARGET_LINK_LIBRARIES(cctUpdate custinger cuUtils cuUpdate cuAlgCCT /usr/local/cuda/lib64/libcudart.so)

#LIST(APPEND CMAKE_CXX_FLAGS "std=c++11 -O3 -ffastmath -Wall")

#LIST(APPEND CUDA_NCCC_FLAGS --compiler-options -fno-strict-aliasing -use-fast-math)
LIST(APPEND CUDA_NCCC_FLAGS --compiler-options)
LIST(APPEND CUDA NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA NVCC_FLAGS "-std=c++11;-DVERBOSE")


if(CMAKE_COMPILER_IS_GNUCXX)
   execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
   if (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
        message(STATUS "C++11 activated.")
        add_definitions("-std=gnu++11")
#        add_definitions("-Wall")
        add_definitions("-Wno-cpp")


   elseif(GCC_VERSION VERSION_GREATER 4.3 OR GCC_VERSION VERSION_EQUAL 4.3)
        message(WARNING "C++0x activated. If you get any errors update to a compiler which fully supports C++11")
        add_definitions("-std=gnu++0x")
   else ()
        message(FATAL_ERROR "C++11 needed. Therefore a gcc compiler with a version higher than 4.3 is needed.")   
   endif()
else(CMAKE_COMPILER_IS_GNUCXX)
   add_definitions("-std=c++0x") 
endif()
