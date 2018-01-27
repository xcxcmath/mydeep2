cmake_minimum_required(VERSION 3.9)
project(mydeep2)

#version
set(PROJECT_VERSION_MAJOR 3)
set(PROJECT_VERSION_MINOR 1)

#Build comfiguration
include_directories(Eigen)
add_compile_options(-W -Wall -O2)

#config for debug
set(CMAKE_CXX_FLAGS_DEBUG "-g")

#Standard C++
set(CMAKE_CXX_STANDARD 17)

#src files
set(core_sources Layer.cpp)
set(core_headers Core.h Layer.h)

set(sources ${core_sources})
set(headers ${core_headers})

add_executable(mydeep2 main.cpp ${sources} ${headers})