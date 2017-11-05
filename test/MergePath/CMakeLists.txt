set(PROJECT "MERGE_PATH")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR  ../../)
#set(CUDA_C_COMPILER
#    "/nethome/fbusata3/bin/llvm-3.6.2.src/build/bin/clang++")
set(WARNING_FLAGS ON)

set(CUB_LIBRARY     "/nethome/fbusata3/lib/cub-1.6.4")

include(${XLIB_DIR}/CMakeLists.txt)
# ------------------------------------------------------------------------------
set(CUDA_SEPARABLE_COMPILATION ON)



cuda_add_executable(merge ${PROJECT_SOURCE_DIR}/main.cu
${XLIB_HOST_SRC} ${GRAPH_HOST_SRC} ${XLIB_DEVICE_SRC} OPTIONS -w)
