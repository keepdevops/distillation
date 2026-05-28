cmake_minimum_required(VERSION 3.18)
project(distill_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── pybind11 ─────────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Prefer pybind11 installed via pip/conda; fall back to FetchContent
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    thermal_bindings.cpp
    llama_bindings.cpp
    metric_streamer.cpp
    lora_bindings.cpp
    export_bindings.cpp
    pybind_module.cpp
)

# ── Module ────────────────────────────────────────────────────────────────────
pybind11_add_module(distill_cpp ${SOURCES})

target_compile_options(distill_cpp PRIVATE
    -O3
    -march=native
    -fvisibility=hidden
    -Wall
    -Wextra
)

# Apple Silicon: link against Accelerate for BLAS
if(APPLE)
    find_library(ACCELERATE_LIB Accelerate REQUIRED)
    target_link_libraries(distill_cpp PRIVATE ${ACCELERATE_LIB})
    target_compile_definitions(distill_cpp PRIVATE PLATFORM_APPLE_SILICON=1)
endif()

# ── Install next to the Python package ───────────────────────────────────────
install(
    TARGETS distill_cpp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../distill/cpp
)

message(STATUS "distill_cpp: Python ${Python3_VERSION}, pybind11 found")
