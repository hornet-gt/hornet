set(PROJECT "XLIB")
project(${PROJECT})

cmake_minimum_required(VERSION 3.5)

#add_compile_options(-stdlib=libc++)
#set(CMAKE_C_COMPILER /nethome/fbusata3/bin/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang)
#set(CMAKE_CXX_COMPILER /nethome/fbusata3/bin/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++)
include(${PROJECT_SOURCE_DIR}/../../CMakeLists.txt)

add_executable(graph ${PROJECT_SOURCE_DIR}/graph.cpp ${XLIB_HOST_SRC} ${GRAPH_HOST_SRC})
