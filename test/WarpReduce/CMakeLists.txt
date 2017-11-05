set(PROJECT "warp_reduce")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR  ../../)

include(${XLIB_DIR}/CMakeBase/CMakeLists.txt)
# ------------------------------------------------------------------------------

#include_directories(${XLIB_DIR})

# ------------------------------------------------------------------------------
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(prova ${PROJECT_SOURCE_DIR}/prova.cu ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})
