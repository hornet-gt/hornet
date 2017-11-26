set(PROJECT "TILE")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR  ../../)
#set(WARNING_FLAGS ON)

set(QUIET_SUBMODULE ON)
set(CUDA_SEPARABLE_COMPILATION ON)
include(${XLIB_DIR}/util/CMakeLists.txt)
# ------------------------------------------------------------------------------
file(GLOB_RECURSE CU_SRCS  ${XLIB_DIR}/src/*.cpp)
include_directories(${XLIB_DIR}/include)

cuda_add_executable(tile_test ${PROJECT_SOURCE_DIR}/main.cu ${CU_SRCS})

target_link_libraries(tile_test ${CUDA_CUBLAS_LIBRARIES})
