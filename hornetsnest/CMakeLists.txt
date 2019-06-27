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

file(GLOB_RECURSE DUMMY         ${PROJECT_SOURCE_DIR}/src/Static/Dummy/Dummy.cu)
file(GLOB_RECURSE BFS_SRCS      ${PROJECT_SOURCE_DIR}/src/Static/BreadthFirstSearch/TopDown.cu)
file(GLOB_RECURSE BFS2_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/BreadthFirstSearch/TopDown2.cu)
file(GLOB_RECURSE BC_SRCS       ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/bc.cu)
file(GLOB_RECURSE BC_SRCS2      ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/approximate_bc.cu)
file(GLOB_RECURSE BC_SRCS3      ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/exact_bc.cu)
file(GLOB_RECURSE BUBFS_SRC     ${PROJECT_SOURCE_DIR}/src/Static/BottomUpBreadthFirstSearch/BottomUpBFS.cu)
file(GLOB_RECURSE CC_SRCS       ${PROJECT_SOURCE_DIR}/src/Static/ConnectedComponents/CC.cu)
file(GLOB_RECURSE CLCOEFF_SRCS  ${PROJECT_SOURCE_DIR}/src/Static/ClusteringCoefficient/cc.cu)
file(GLOB_RECURSE SSSP_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/ShortestPath/SSSP.cu)
file(GLOB_RECURSE SPMV_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/SpMV/SpMV.cu)
file(GLOB_RECURSE PR_SRCS       ${PROJECT_SOURCE_DIR}/src/Static/PageRank/PageRank.cu)
file(GLOB_RECURSE KCORE_SRCS    ${PROJECT_SOURCE_DIR}/src/Static/KCore/KCore.cu)
file(GLOB_RECURSE KATZ_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/KatzCentrality/Katz.cu)
file(GLOB_RECURSE TRI2_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/TriangleCounting/triangle2.cu)
file(GLOB_RECURSE X_SRCS        ${PROJECT_SOURCE_DIR}/../xlib/src/*)
file(GLOB_RECURSE H_SRCS        ${PROJECT_SOURCE_DIR}/../hornet/src/*)

add_library(hornetAlg ${X_SRCS} ${H_SRCS} ${DUMMY} ${BFS_SRCS} ${BFS2_SRCS} ${BC_SRCS} ${BC_SRCS2} ${BC_SRCS3} ${BUBFS_SRC} ${CC_SRCS} ${CLCOEFF_SRCS} ${SSSP_SRCS} ${SPMV_SRCS} ${PR_SRCS} ${KATZ_SRCS} ${KCORE_SRCS} ${TRI2_SRCS})

###################################################################################################
# - add executables -------------------------------------------------------------------------------

add_executable(dummy        test/DummyTest.cu)
add_executable(bfs          test/BFSTest.cu)
add_executable(bfs2         test/BFSTest2.cu)
add_executable(bc	          test/BCTest.cu)
add_executable(bubfs        test/BUBFSTest2.cu)
add_executable(con-comp     test/CCTest.cu)
add_executable(spmv         test/SpMVTest.cu)
add_executable(sssp         test/SSSPTest.cu)
add_executable(katz         test/KatzTest.cu)
add_executable(kcore        test/KCoreTest.cu)
#add_executable(dyn-katz     test/KatzDynamicTest.cu)
add_executable(triangle2    test/TriangleTest2.cu)
add_executable(clus-coeff   test/ClusCoeffTest.cu)
add_executable(pr           test/PageRankTest.cu)

target_link_libraries(dummy         hornetAlg)
target_link_libraries(bfs           hornetAlg)
target_link_libraries(bfs2          hornetAlg)
target_link_libraries(bc            hornetAlg)
target_link_libraries(bubfs         hornetAlg)
target_link_libraries(con-comp      hornetAlg)
target_link_libraries(spmv          hornetAlg)
target_link_libraries(sssp          hornetAlg)
target_link_libraries(katz          hornetAlg)
target_link_libraries(kcore         hornetAlg)
#target_link_libraries(dyn-katz      hornetAlg)
target_link_libraries(triangle2     hornetAlg)
target_link_libraries(clus-coeff    hornetAlg)
target_link_libraries(pr            hornetAlg)
