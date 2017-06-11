get_directory_property(has_parent PARENT_DIRECTORY)
if (has_parent)
    set(ENABLE_TARGET OFF)
else()
    set(ENABLE_TARGET ON)
endif()
#-------------------------------------------------------------------------------
set(PROJECT "cuSTINGER")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

#set(WARNING_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
include_directories(include)
include_directories(externals/cub)
include_directories(${CUDA_INCLUDE_DIRS})

include(util/CMakeLists.txt)
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)

cuda_add_library(cuStinger ${CPP_SRCS} ${CU_SRCS})
#-------------------------------------------------------------------------------

cuda_add_executable(cuStingerTest test/cuStingerTest.cpp)
target_link_libraries(cuStingerTest cuStinger ${CUDA_LIBRARIES})

#cuda_add_executable(mem_test test/MemoryManagement.cu)
#TARGET_LINK_LIBRARIES(mem_test cuStinger)
#-------------------------------------------------------------------------------
##subscript_out_of_range
