set(PROJECT "cuStingerAlg")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

#set(WARNING_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
include(externals/cuStinger/util/CMakeLists.txt)
include_directories(include)
include_directories(externals/cuStinger/include)
include_directories(${CUDA_INCLUDE_DIRS})
#-------------------------------------------------------------------------------
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)

set(cuStingerLibraryDir
    ${PROJECT_SOURCE_DIR}/externals/cuStinger/build
    CACHE PATH "Directory to the cuSTINGER library")

link_directories(${cuStingerLibraryDir})

#-------------------------------------------------------------------------------

cuda_add_executable(bfs++ test/BFS++.cu ${CU_SRCS})
cuda_add_executable(bfs   test/BFS.cu ${CU_SRCS})

target_link_libraries(bfs++ cuStinger ${CUDA_LIBRARIES})
target_link_libraries(bfs   cuStinger ${CUDA_LIBRARIES})
