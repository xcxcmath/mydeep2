cmake_minimum_required(VERSION 3.9)
project(mydeep2)

#version
set(PROJECT_VERSION_MAJOR 3)
set(PROJECT_VERSION_MINOR 1)

#Build configuration
include_directories(Eigen)
add_compile_options(-W -Wall -O2)

#config for debug
set(CMAKE_CXX_FLAGS_DEBUG "-g")

#Standard C++
set(CMAKE_CXX_STANDARD 17)

#src files
set(core_sources)
set(core_headers Common.h)
set(layer_sources Layer.cpp Hidden.cpp Affine.cpp Activation.cpp Dropout.cpp BatchNorm.cpp Output.cpp)
set(layer_headers Layer.h Hidden.h Affine.h Activation.h Dropout.h BatchNorm.h Output.h)
set(network_sources Network.cpp)
set(network_headers Network.h)

set(optimizer_sources Optimizer.cpp Momentum.cpp Adam.cpp NAG.cpp)
set(optimizer_headers Optimizer.h Momentum.h Adam.h NAG.h)

set(sources ${core_sources}
            ${layer_sources}
        ${network_sources}
        ${optimizer_sources})
set(headers ${core_headers}
            ${layer_headers}
        ${network_headers}
        ${optimizer_headers} Dense.h)

set(src ${sources} ${headers})

add_library(mydeep2lib ${src})
add_executable(mydeep2 main.cpp ${src})
target_link_libraries(mydeep2 mydeep2lib)