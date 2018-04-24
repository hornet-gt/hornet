get_directory_property(has_parent PARENT_DIRECTORY)
if (has_parent)
    set(QUIET_SUBMODULE ON)
else()
    set(QUIET_SUBMODULE OFF)
endif()

#-------------------------------------------------------------------------------
set(PROJECT "Hornet")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(include)
include_directories(externals/cub-1.8.0)
include_directories(externals/moderngpu/src)
include_directories(externals/cpp-btree-master)
include_directories(externals/xlib/include)
include_directories(${CUDA_INCLUDE_DIRS})

#set(WARNING_FLAGS ON)
set(CUB_LIBRARY ON)
include(externals/xlib/util/CMakeLists.txt)

#-------------------------------------------------------------------------------
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE SCPP_SRCS ${PROJECT_SOURCE_DIR}/externals/xlib/src/*.cpp)
file(GLOB_RECURSE SCU_SRCS  ${PROJECT_SOURCE_DIR}/externals/xlib/src/*.cu)

cuda_add_library(hornet ${CPP_SRCS} ${CU_SRCS} ${SCPP_SRCS} ${SCU_SRCS})
#-------------------------------------------------------------------------------

cuda_add_executable(lb_test       test/BinarySearchTest.cu)
cuda_add_executable(ptxtest       test/ptxtest.cu)
cuda_add_executable(hornet_test   test/HornetTest.cu)
cuda_add_executable(hornet_insert_test   test/HornetInsertTest.cu)
cuda_add_executable(hornet_delete_test   test/HornetDeleteTest.cu)
cuda_add_executable(mem_benchmark test/MemBenchmark.cu)
cuda_add_executable(weight_test   test/WeightTest.cu)

target_link_libraries(lb_test       hornet ${CUDA_LIBRARIES})
target_link_libraries(ptxtest       hornet ${CUDA_LIBRARIES})
target_link_libraries(hornet_test   hornet ${CUDA_LIBRARIES})
target_link_libraries(hornet_insert_test   hornet ${CUDA_LIBRARIES})
target_link_libraries(hornet_delete_test   hornet ${CUDA_LIBRARIES})
target_link_libraries(mem_benchmark hornet ${CUDA_LIBRARIES})
target_link_libraries(weight_test   hornet ${CUDA_LIBRARIES})

#cuda_add_executable(data_layout test/DataLayout.cu)
#cuda_add_executable(csr_test test/CsrTest.cu)
#target_link_libraries(csr_test hornet ${CUDA_LIBRARIES})
#target_link_libraries(data_layout hornet ${CUDA_LIBRARIES})
#cuda_add_executable(mem_test test/MemoryManagement.cu)
#TARGET_LINK_LIBRARIES(mem_test hornet)
#-------------------------------------------------------------------------------
##subscript_out_of_range
#cmake -DCMAKE_BUILD_TYPE=Release -DCUDAHC=$clang39 ..
