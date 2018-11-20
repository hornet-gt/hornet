set(PROJECT "XLIB")
project(${PROJECT})

cmake_minimum_required(VERSION 3.5)
#FIND_PACKAGE(CUDA REQUIRED)

set(WARNING_FLAGS ON)

include(${PROJECT_SOURCE_DIR}/../CMakeLists.txt)

add_library(XLib ${XLIB_HOST_SRC} ${GRAPH_HOST_SRC_EXP})

# ------------------------------------------------------------------------------
#set(CUDA_SEPARABLE_COMPILATION ON)

#CUDA_ADD_LIBRARY(XLib ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC}
#                        ${GRAPH_HOST_SRC} ${GRAPH_DEVICE_SRC})


if (${TEST})

cuda_add_executable(ThreadReduceTest
                    ${PROJECT_SOURCE_DIR}/test/Primitives/ThreadReduceTest.cu
                    ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})

#cuda_add_executable(ThreadScanTest
#                    ${PROJECT_SOURCE_DIR}/test/Primitives/ThreadScanTest.cu
#                    ${XLIB_HOST_SRC})

cuda_add_executable(CacheModifiersTest
                    ${PROJECT_SOURCE_DIR}/test/DataMovement/CacheModifiersTest.cu
                    ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})

cuda_add_executable(PTXTest
                    ${PROJECT_SOURCE_DIR}/test/PTX/PTXtest.cu
                ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC} OPTIONS -keep --resource-usage)

enable_testing()

add_test(NAME ThreadReduceTest
         COMMAND ThreadReduceTest)
#add_test(NAME ThreadScanTest
#         COMMAND ThreadScanTest)
add_test(NAME CacheModifiersTest
         COMMAND CacheModifiersTest)
add_test(NAME PTXTest
         COMMAND PTXTest)

endif()
