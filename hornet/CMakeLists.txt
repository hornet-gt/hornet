cmake_minimum_required(VERSION 3.8) #language level CUDA support was added in CMake 3.8

set(PROJECT "Hornet")
project(${PROJECT} LANGUAGES CXX CUDA)

include(../compiler-util/CMakeLists.txt)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(include)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

if(RMM_LIBRARY)
    add_library(rmm SHARED IMPORTED)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/../externals/rmm/build/librmm.so)
endif()

file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE SCPP_SRCS ${PROJECT_SOURCE_DIR}/../xlib/src/*.cpp)
file(GLOB_RECURSE SCU_SRCS  ${PROJECT_SOURCE_DIR}/../xlib/src/*.cu)

add_library(hornet ${CPP_SRCS} ${CU_SRCS} ${SCPP_SRCS} ${SCU_SRCS})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

if(RMM_LIBRARY)
target_link_libraries(hornet rmm)
endif()
if(USE_NVTX)
target_link_libraries(hornet ${NVTX_LIBRARIES})
endif()

###################################################################################################
# - add executables -------------------------------------------------------------------------------

add_executable(lb_test                            test/BinarySearchTest.cu)
add_executable(ptxtest                            test/ptxtest.cu)
add_executable(hornet_mgpu_insert_test            test/HornetMultiGPUInsertTest.cu)
add_executable(hornet_insert_weighted_test        test/HornetInsertTestWeighted.cu)
add_executable(hornet_insert_test                 test/HornetInsertTest.cu)
add_executable(hornet_delete_test                 test/HornetDeleteTest.cu)
add_executable(mem_benchmark                      test/MemBenchmark.cu)
if(RMM_LIBRARY)
add_executable(rmm_benchmark      test/RMMBenchmark.cu)
endif()

target_link_libraries(lb_test                           hornet)
target_link_libraries(ptxtest                           hornet)
target_link_libraries(hornet_mgpu_insert_test           hornet)
target_link_libraries(hornet_insert_weighted_test       hornet)
target_link_libraries(hornet_insert_test                hornet)
target_link_libraries(hornet_delete_test                hornet)
target_link_libraries(mem_benchmark                     hornet)
if(RMM_LIBRARY)
target_link_libraries(rmm_benchmark      hornet)
endif()

