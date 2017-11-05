set(PROJECT "Oversize")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR  ../../)

include(${XLIB_DIR}/CMakeLists.txt)
# ------------------------------------------------------------------------------
include_directories(${XLIB_DIR}/../lib/cub-1.6.4)
include_directories(${XLIB_DIR}/include)

# ------------------------------------------------------------------------------
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(duplicate ${PROJECT_SOURCE_DIR}/main.cu
                              ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})
