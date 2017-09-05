set(PROJECT "HornetAlg")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

add_subdirectory(externals/cuStinger)

include_directories(include)
include_directories(externals/cuStinger/include)
include_directories(externals/cuStinger/externals/cub-1.7.3)
include_directories(externals/cuStinger/externals/cpp-btree-master)
include_directories(externals/cuStinger/externals/Support/include)

set(QUIET_SUBMODULE ON)
include(externals/cuStinger/util/CMakeLists.txt)
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CU_SRCS ${PROJECT_SOURCE_DIR}/src/Static/*.cu)

cuda_add_library(hornetAlg ${CU_SRCS} OPTIONS ${CSR_FLAG})
#-------------------------------------------------------------------------------

cuda_add_executable(bfs test/BFSTest.cu OPTIONS ${CSR_FLAG})
#cuda_add_executable(bfs   test/BFSTest.cu OPTIONS ${CSR_FLAG})
#cuda_add_executable(CC    test/CCTest.cu)
#cuda_add_executable(KTruss test/KTrusstest.cu)
#cuda_add_executable(Triangle test/TriangleTest.cu)

target_link_libraries(bfs hornetAlg hornet)
#target_link_libraries(bfs   HornetAlg cuStinger)
#target_link_libraries(CC    HornetAlg cuStinger)
#target_link_libraries(KTruss HornetAlg cuStinger)
#target_link_libraries(Triangle HornetAlg cuStinger)
