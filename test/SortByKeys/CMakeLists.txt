set(PROJECT "MERGE_PATH")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)

set(XLIB_DIR  ../../)
set(WARNING_FLAGS ON)

include(${XLIB_DIR}/CMakeLists.txt)
# ------------------------------------------------------------------------------

add_executable(merge ${PROJECT_SOURCE_DIR}/main.cpp
                ${XLIB_HOST_SRC})
