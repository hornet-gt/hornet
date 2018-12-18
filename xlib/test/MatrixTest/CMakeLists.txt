set(PROJECT "Matrix")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)

set(XLIB_DIR  ../../)

include(${XLIB_DIR}/CMakeBase/CMakeLists.txt)

add_executable(prova ${PROJECT_SOURCE_DIR}/prova.cpp ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})
