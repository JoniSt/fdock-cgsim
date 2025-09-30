cmake_minimum_required(VERSION 3.20)

# Project setup
project(fdock_cpu LANGUAGES C CXX)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enforce C++23 and common warnings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(fdock_cpu)

target_sources(fdock_cpu
  PRIVATE
    # C sources
    fdock.c
    getparameters.c
    miscellaneous.c
    performdocking.c
    processgrid.c
    processligand.c
    processresult.c
    searchoptimum.c
    # C++ sources
    processligand_graphtoy.cpp
    graphtoy/graphtoy.cpp
    # Headers (for IDE visibility)
    defines.h
    getparameters.h
    miscellaneous.h
    performdocking.h
    processgrid.h
    processligand.h
    processresult.h
    searchoptimum.h
    json.hpp
    graphtoy/graphtoy.hpp
)

target_include_directories(fdock_cpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/graphtoy
)

target_compile_options(fdock_cpu PRIVATE -Wall)

# Link libm explicitly
target_link_libraries(fdock_cpu PRIVATE m)
