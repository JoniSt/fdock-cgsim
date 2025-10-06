cmake_minimum_required(VERSION 3.20)

# Project setup
project(fdock_cpu LANGUAGES C CXX)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Workaround for Clang tool being unable to find the standard headers if GCC stdlib is used
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_executable(fdock_cpu)

target_sources(fdock_cpu PRIVATE
    fdock.c
    getparameters.c
    miscellaneous.c
    performdocking.c
    processgrid.c
    processligand.c
    processresult.c
    searchoptimum.c
    processligand_graphtoy.cpp

    defines.h
    getparameters.h
    miscellaneous.h
    miscellaneous_inline.h
    performdocking.h
    processgrid.h
    processligand.h
    processresult.h
    searchoptimum.h
    json.hpp
)

if (NOT DEFINED ENV{CGSIM_INSTALL_DIR})
    message(FATAL_ERROR "CGSIM_INSTALL_DIR environment variable not set")
endif()

target_include_directories(fdock_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/graphtoy
    $ENV{CGSIM_INSTALL_DIR}/include
)

target_link_directories(fdock_cpu PRIVATE
    $ENV{CGSIM_INSTALL_DIR}/lib
)

target_compile_options(fdock_cpu PRIVATE -Wall)
target_link_libraries(fdock_cpu PRIVATE m cgsim)

add_executable(gen_fdock_luts tools/gen_fdock_luts.cpp)
target_include_directories(gen_fdock_luts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/fdock_intraE_luts.hpp)

add_custom_command(
    OUTPUT ${GENERATED_HEADER}
    COMMAND gen_fdock_luts > ${GENERATED_HEADER}
    DEPENDS gen_fdock_luts
    COMMENT "Generating IntraE LUT header"
    VERBATIM
)

add_custom_target(fdock_intraE_luts_header ALL
    DEPENDS ${GENERATED_HEADER}
)

add_dependencies(fdock_cpu fdock_intraE_luts_header)

target_include_directories(fdock_cpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
