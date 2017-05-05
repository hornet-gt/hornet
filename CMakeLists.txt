set(PROJECT "cuStingerAlg")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

add_subdirectory(externals/cuStinger)

include_directories(include)
include_directories(externals/cuStinger/externals/cub)
include_directories(externals/cuStinger/include)
include(externals/cuStinger/util/CMakeLists.txt)
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)

cuda_add_executable(bfs++ test/BFStest++.cu ${CU_SRCS})
cuda_add_executable(bfs   test/BFStest.cu ${CU_SRCS})

target_link_libraries(bfs++ cuStinger ${CUDA_LIBRARIES})
target_link_libraries(bfs   cuStinger ${CUDA_LIBRARIES})
