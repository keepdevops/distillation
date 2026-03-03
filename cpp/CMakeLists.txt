cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Distillation)

set(CMAKE_CXX_STANDARD 17)

# LibTorch distillation: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -B build && cmake --build build
set(CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}" CACHE PATH "Path to LibTorch")
find_package(Torch QUIET)
if(Torch_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  add_executable(distill main.cpp)
  target_link_libraries(distill "${TORCH_LIBRARIES}")
endif()

# Watchdog: standalone, no LibTorch. Uses vendored nlohmann/json single header.
add_executable(watchdog watchdog_main.cpp)
target_include_directories(watchdog PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
