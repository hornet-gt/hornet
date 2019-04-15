cmake_minimum_required(VERSION 3.8) #language level CUDA support was added in CMake 3.8

set(PROJECT "HornetAlg")
project(${PROJECT} LANGUAGES CXX CUDA)

include(../compiler-util/CMakeLists.txt)

#add_subdirectory(../hornet build)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(../hornet/include)
include_directories(include)
include_directories(../primitives)
include_directories(${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

file(GLOB_RECURSE DUMMY ${PROJECT_SOURCE_DIR}/src/Static/Dummy/Dummy.cu)
file(GLOB_RECURSE BFS_SRCS ${PROJECT_SOURCE_DIR}/src/Static/BreadthFirstSearch/TopDown.cu)
file(GLOB_RECURSE BFS2_SRCS ${PROJECT_SOURCE_DIR}/src/Static/BreadthFirstSearch/TopDown2.cu)
file(GLOB_RECURSE BC_SRCS ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/bc.cu)
file(GLOB_RECURSE BC_SRCS2 ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/approximate_bc.cu)
file(GLOB_RECURSE BC_SRCS3 ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/exact_bc.cu)
file(GLOB_RECURSE CC_SRCS ${PROJECT_SOURCE_DIR}/src/Static/ConnectedComponents/CC.cu)
file(GLOB_RECURSE PR_SRCS ${PROJECT_SOURCE_DIR}/src/Static/PageRank/PageRank.cu)
file(GLOB_RECURSE TRI2_SRCS ${PROJECT_SOURCE_DIR}/src/Static/TriangleCounting/triangle2.cu)
file(GLOB_RECURSE X_SRCS   ${PROJECT_SOURCE_DIR}/../xlib/src/*)
file(GLOB_RECURSE H_SRCS   ${PROJECT_SOURCE_DIR}/../hornet/src/*)

add_library(hornetAlg ${X_SRCS} ${H_SRCS} ${DUMMY} ${BFS_SRCS} ${BFS2_SRCS} ${BC_SRCS} ${BC_SRCS2} ${BC_SRCS3} ${CC_SRCS} ${PR_SRCS} ${TRI2_SRCS})

###################################################################################################
# - add executables -------------------------------------------------------------------------------

add_executable(p           test/DummyTest.cu)
add_executable(bfs         test/BFSTest.cu)
add_executable(bfs2        test/BFSTest2.cu)
add_executable(bc	   test/BCTest.cu)
add_executable(con-comp   test/CCTest.cu)
#add_executable(spmv       test/SpMVTest.cu)
#add_executable(sssp       test/SSSPTest.cu)
#add_executable(katz       test/KatzTest.cu)
#add_executable(dyn-katz   test/KatzDynamicTest.cu)
#add_executable(triangle   test/TriangleTest.cu)
add_executable(triangle2  test/TriangleTest2.cu)
#add_executable(clus-coeff test/ClusCoeffTest.cu)
add_executable(pr          test/PageRankTest.cu)

target_link_libraries(p           hornetAlg)
target_link_libraries(bfs         hornetAlg)
target_link_libraries(bfs2        hornetAlg)
target_link_libraries(bc	  hornetAlg)
target_link_libraries(con-comp    hornetAlg)
#target_link_libraries(spmv       hornetAlg hornet)
#target_link_libraries(sssp       hornetAlg hornet)
#target_link_libraries(katz       hornetAlg hornet)
#target_link_libraries(dyn-katz   hornetAlg hornet)
#target_link_libraries(triangle   hornetAlg hornet)
target_link_libraries(triangle2  hornetAlg)
#target_link_libraries(clus-coeff hornetAlg hornet)
target_link_libraries(pr          hornetAlg)

