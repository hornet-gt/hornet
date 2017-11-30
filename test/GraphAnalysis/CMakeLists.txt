set(PROJECT "GraphAnalysis")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)

set(XLIB_DIR  ../../)
#set(WARNING_FLAGS ON)
#set(QUIET_SUBMODULE ON)

include(${XLIB_DIR}/util/CMakeLists.txt)
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SRCS ${XLIB_DIR}/src/Host/*.cpp
                       ${XLIB_DIR}/src/Graph/*.cpp)
include_directories(${XLIB_DIR}/include)

add_executable(graph_analysis ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
