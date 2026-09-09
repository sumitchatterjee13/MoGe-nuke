cmake_minimum_required(VERSION 3.20)
project(MoGe3OFX CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MoGe3 SHARED src/moge_ofx.cpp)
target_include_directories(MoGe3 PRIVATE include/openfx)
target_link_libraries(MoGe3 PRIVATE ws2_32)
if(MSVC)
  target_compile_options(MoGe3 PRIVATE /W3 /O2 /utf-8)
endif()

# OFX bundle layout: MoGe3.ofx.bundle/Contents/Win64/MoGe3.ofx
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/MoGe3.ofx.bundle/Contents/Win64")
set_target_properties(MoGe3 PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  RUNTIME_OUTPUT_DIRECTORY "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BUNDLE_DIR}")
