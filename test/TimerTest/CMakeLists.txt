set(PROJECT "timer_test")
project(${PROJECT} C CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

set(XLIB_DIR  ../../)

set(WARNING_FLAGS ON)
include(${XLIB_DIR}/CMakeBase/CMakeLists.txt)
# ------------------------------------------------------------------------------

#include_directories(${XLIB_DIR})

# ------------------------------------------------------------------------------
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(prova ${PROJECT_SOURCE_DIR}/prova.cu ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})

#add_executable(pp ${PROJECT_SOURCE_DIR}/prova.cpp ${PROJECT_SOURCE_DIR}/prova.cuh)
#add_executable(pp ${PROJECT_SOURCE_DIR}/prova.cu ${PROJECT_SOURCE_DIR}/prova.cuh)
#add_executable(prova ${PROJECT_SOURCE_DIR}/prova.cu)
#set_target_properties(pp PROPERTIES LINKER_LANGUAGE CXX)
