set(PROJECT "Thread Reduce")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR  ../../)

include(${XLIB_DIR}/CMakeBase/CMakeLists.txt)
# ------------------------------------------------------------------------------

include_directories(${XLIB_DIR}/include)

# ------------------------------------------------------------------------------
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(reduceTest ${PROJECT_SOURCE_DIR}/ThreadReduceTest.cu
                             ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})
