set(PROJECT "OptimizedFill")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR    ../../)

include(${XLIB_DIR}/CMakeBase/CMakeLists.txt)
# ------------------------------------------------------------------------------

include_directories(${XLIB_DIR}
                     ${XLIB_DIR}/../cub-1.4.1/
)

set(UTIL_SRC     ${XLIB_DIR}/Base/Host/src/fUtil.cpp
                ${XLIB_DIR}/Base/Host/src/Timer.cpp
                ${XLIB_DIR}/Base/Device/Util/src/Timer.cu
                ${XLIB_DIR}/Base/Device/Util/src/cuda_util.cu
)

# ------------------------------------------------------------------------------

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -keep -keep-dir=${PROJECT_SOURCE_DIR}/build/TMP")

cuda_add_executable(OptimizedFill ${PROJECT_SOURCE_DIR}/OptimizedFill.cu ${UTIL_SRC})
