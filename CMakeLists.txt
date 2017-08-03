set(PROJECT "cuStingerAlg")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

add_subdirectory(externals/cuStinger)

include_directories(include)
include_directories(externals/cuStinger/externals/cub)
include_directories(externals/cuStinger/include)

set(ENABLE_TARGET ON)
include(externals/cuStinger/util/CMakeLists.txt)
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CU_SRCS ${PROJECT_SOURCE_DIR}/src/Static/*.cu)
cuda_add_library(cuStingerAlg ${CU_SRCS})
#-------------------------------------------------------------------------------

cuda_add_executable(bfs++ test/BFSTest++.cu)
cuda_add_executable(bfs   test/BFSTest.cu)
#cuda_add_executable(CC    test/CCTest.cu)
#cuda_add_executable(KTruss test/KTrusstest.cu)
cuda_add_executable(katz   test/KatzTest.cu)

target_link_libraries(bfs++ cuStingerAlg cuStinger)
target_link_libraries(bfs   cuStingerAlg cuStinger)
#target_link_libraries(CC    cuStingerAlg cuStinger)
#target_link_libraries(KTruss cuStingerAlg cuStinger)
target_link_libraries(katz cuStingerAlg cuStinger)
