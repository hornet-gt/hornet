set(PROJECT "cuSTINGER")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

#set(WARNING_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
include(util/CMakeLists.txt)
include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(externals/stx-btree/include)
#include_directories(externals/moderngpu/src/moderngpu)
#include_directories(externals/cub/cub/)
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)

cuda_add_library(cuStinger ${CPP_SRCS} ${CU_SRCS})

cuda_add_executable(cuStingerTest test/cuStingerTest.cpp)
#cuda_add_executable(mem_test test/MemoryManagement.cu)
cuda_add_executable(bfs test/BFS.cu ${CU_SRCS})

target_link_libraries(cuStingerTest cuStinger ${CUDA_LIBRARIES})
target_link_libraries(bfs           cuStinger ${CUDA_LIBRARIES})

#TARGET_LINK_LIBRARIES(mem_test cuStinger)
#-------------------------------------------------------------------------------
##subscript_out_of_range
