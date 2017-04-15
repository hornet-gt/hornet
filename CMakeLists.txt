set(PROJECT "cuSTINGER")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

#set(WARNING_FLAGS ON)
include(util/CMakeLists.txt)
include_directories(include)
include_directories(externals/stx-btree/include)
include_directories(externals/moderngpu/src/moderngpu)
include_directories(externals/cub/cub/)
include_directories(${CUDA_INCLUDE_DIRS})
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)

cuda_add_library(cuStinger ${CPP_SRCS} ${CU_SRCS})

cuda_add_executable(cuStingerTest src/main.cpp)
#cuda_add_executable(mem_test test/MemoryManagement.cu)

TARGET_LINK_LIBRARIES(cuStingerTest cuStinger ${CUDA_LIBRARIES})
#TARGET_LINK_LIBRARIES(mem_test cuStinger)
#-------------------------------------------------------------------------------

#set(mgpu_INCLUDE_DIRS
#  ${CMAKE_SOURCE_DIR}/externals/moderngpu/include
#  CACHE PATH
#  "Directory to the Modern GPU include files")

#set(mgpu_SOURCE_DIRS
#  ${CMAKE_SOURCE_DIR}/externals/moderngpu/src
#  CACHE PATH
 # "Directory to the Modern GPU source files")

# Adding external libraries
#include_directories(externals/stx-btree/include)
#include_directories(externals/moderngpu/src/moderngpu)
#include_directories(externals/cub/cub/)
