cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(Talulla VERSION 0.1
    LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_CXX_FLAGS "-ansi -pedantic-errors -Wall -Wextra")
set(CMAKE_CXX_FLAGS "-ansi -Wall -Wextra")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#set(CMAKE_VERBOSE_MAKEFILE ON)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

find_package(CUDA 10.1 REQUIRED)

set(SOURCE_FILES
    src/talulla.cpp
    src/tensors.cpp
    src/cuda_helper.cpp
    src/dropout.cpp
    src/mmio_wrapper.cpp
    src/mmio.c
    src/sparse_computation.cpp
    src/dense_computation.cpp
    src/convolution.cpp)

add_executable(${PROJECT_NAME}
    ${SOURCE_FILES})
target_link_directories(${PROJECT_NAME}
    PRIVATE /usr/local/lib)
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
    cnpy
    cudnn
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_CUBLAS_LIBRARIES})
