set(PROJECT "iterator")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR  ../../)

include(${XLIB_DIR}/CMakeBase/CMakeLists.txt)
# ------------------------------------------------------------------------------

include_directories(${XLIB_DIR}
                     ${XLIB_DIR}/../cub-1.4.1/
)

# ------------------------------------------------------------------------------
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(Iterator ${PROJECT_SOURCE_DIR}/Iterator.cu ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})
