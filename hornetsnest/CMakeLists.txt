cmake_minimum_required(VERSION 3.8) #language level CUDA support was added in CMake 3.8

set(PROJECT "HornetAlg")
project(${PROJECT} LANGUAGES CXX CUDA)

include(../compiler-util/CMakeLists.txt)

add_subdirectory(../hornet build)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(../hornet/include)
include_directories(include)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

file(GLOB_RECURSE CU_SRCS ${PROJECT_SOURCE_DIR}/src/*)
list(REMOVE_ITEM CU_SRCS ${PROJECT_SOURCE_DIR}/src/Dynamic/KatzCentrality/Katz.cu)
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/externals/xlib/src/*)

add_library(hornetAlg ${CU_SRCS})

###################################################################################################
# - add executables -------------------------------------------------------------------------------

add_executable(bfs        test/BFSTest.cu)
add_executable(bfs2       test/BFSTest2.cu)
add_executable(con-comp   test/CCTest.cu)
add_executable(spmv       test/SpMVTest.cu)
add_executable(sssp       test/SSSPTest.cu)
add_executable(katz       test/KatzTest.cu)
add_executable(dyn-katz   test/KatzDynamicTest.cu)
add_executable(triangle   test/TriangleTest.cu)
add_executable(triangle2  test/TriangleTest2.cu)
add_executable(clus-coeff test/ClusCoeffTest.cu)
add_executable(pr         test/PageRankTest.cu)
add_executable(bc         test/BCTest.cu)
add_executable(kcore      test/KCoreTest.cu)
add_executable(bubfs      test/BUBFSTest2.cu)

target_link_libraries(bfs        hornetAlg hornet)
target_link_libraries(bfs2       hornetAlg hornet)
target_link_libraries(con-comp   hornetAlg hornet)
target_link_libraries(spmv       hornetAlg hornet)
target_link_libraries(sssp       hornetAlg hornet)
target_link_libraries(katz       hornetAlg hornet)
target_link_libraries(dyn-katz   hornetAlg hornet)
target_link_libraries(triangle   hornetAlg hornet)
target_link_libraries(triangle2  hornetAlg hornet)
target_link_libraries(clus-coeff hornetAlg hornet)
target_link_libraries(pr         hornetAlg hornet)
target_link_libraries(bc         hornetAlg hornet)
target_link_libraries(kcore      hornetAlg hornet)
target_link_libraries(bubfs      hornetAlg hornet)
