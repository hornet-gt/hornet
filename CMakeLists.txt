set(PROJECT "HornetAlg")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

add_subdirectory(externals/hornet)

include_directories(include)
include_directories(externals/hornet/include)
include_directories(externals/hornet/externals/cub-1.7.3)
include_directories(externals/hornet/externals/cpp-btree-master)
include_directories(externals/hornet/externals/Support/include)

set(QUIET_SUBMODULE ON)
include(externals/hornet/util/CMakeLists.txt)
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CU_SRCS ${PROJECT_SOURCE_DIR}/src/*.cu)

cuda_add_library(hornetAlg ${CU_SRCS})
#-------------------------------------------------------------------------------
cuda_add_executable(bfs      test/BFSTest.cu)
cuda_add_executable(bfs2     test/BFSTest2.cu)
cuda_add_executable(brim     test/BrimTest.cu)
cuda_add_executable(cc       test/CCTest.cu)
cuda_add_executable(spmv     test/SpMVTest.cu)
cuda_add_executable(sssp     test/SSSPTest.cu)
cuda_add_executable(katz     test/KatzTest.cu)
cuda_add_executable(dyn_katz test/KatzDynamicTest.cu)
cuda_add_executable(ktruss   test/KTrussTest.cu)
#cuda_add_executable(Triangle test/TriangleTest.cu)

target_link_libraries(bfs      hornetAlg hornet)
target_link_libraries(bfs2     hornetAlg hornet)
target_link_libraries(brim     hornetAlg hornet)
target_link_libraries(cc       hornetAlg hornet)
target_link_libraries(spmv     hornetAlg hornet)
target_link_libraries(sssp     hornetAlg hornet)
target_link_libraries(katz     hornetAlg hornet)
target_link_libraries(dyn_katz hornetAlg hornet)
target_link_libraries(ktruss   hornetAlg hornet)
#target_link_libraries(Triangle HornetAlg hornet)
