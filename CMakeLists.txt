get_directory_property(has_parent PARENT_DIRECTORY)
if (has_parent)
    set(QUIET_SUBMODULE OFF)
else()
    set(QUIET_SUBMODULE ON)
endif()

#-------------------------------------------------------------------------------
set(PROJECT "cuSTINGER")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

#set(WARNING_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
include_directories(include)
include_directories(externals/cub-1.7.0)
include_directories(externals/cpp-btree-master)
include_directories(externals/Support/include)
include_directories(${CUDA_INCLUDE_DIRS})

include(util/CMakeLists.txt)
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE SCPP_SRCS ${PROJECT_SOURCE_DIR}/externals/Support/src/*.cpp)
file(GLOB_RECURSE SCU_SRCS  ${PROJECT_SOURCE_DIR}/externals/Support/src/*.cu)

if(CSR)
    set(CSR_FLAG "-DCSR_GRAPH")
    add_compile_options("-DCSR_GRAPH")
    message("==> CSR enabled")
endif()

cuda_add_library(cuStinger ${CPP_SRCS} ${CU_SRCS} ${SCPP_SRCS} ${SCU_SRCS}
                 OPTIONS ${CSR_FLAG})
#-------------------------------------------------------------------------------

#cuda_add_executable(ptxtest test/ptxtest.cu)
cuda_add_executable(cuStingerTest test/cuStingerTest.cpp OPTIONS ${CSR_FLAG})
cuda_add_executable(mem_benchmark test/MemBenchmark.cu   OPTIONS ${CSR_FLAG})

target_link_libraries(cuStingerTest cuStinger ${CUDA_LIBRARIES})
target_link_libraries(mem_benchmark cuStinger ${CUDA_LIBRARIES})

#cuda_add_executable(mem_test test/MemoryManagement.cu)
#TARGET_LINK_LIBRARIES(mem_test cuStinger)
#-------------------------------------------------------------------------------
##subscript_out_of_range
#cmake -DCMAKE_BUILD_TYPE=Release -DCUDAHC=$clang39 ..

