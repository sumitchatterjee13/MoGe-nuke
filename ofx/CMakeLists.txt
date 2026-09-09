cmake_minimum_required(VERSION 3.20)
project(MoGe3OFX CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(MoGe3 SHARED src/moge_ofx.cpp)
target_include_directories(MoGe3 PRIVATE include/openfx)

if(WIN32)
  target_link_libraries(MoGe3 PRIVATE ws2_32)
  set(OFX_ARCH_DIR "Win64")
  if(MSVC)
    target_compile_options(MoGe3 PRIVATE /W3 /O2 /utf-8)
  endif()
else()
  set(OFX_ARCH_DIR "Linux-x86-64")
  target_compile_options(MoGe3 PRIVATE -Wall -O2 -fvisibility=hidden -fPIC)
  target_link_libraries(MoGe3 PRIVATE dl pthread)
  # keep every symbol but the OFX entry points private, and avoid depending
  # on a newer libstdc++ than the host ships
  target_link_options(MoGe3 PRIVATE -Wl,--exclude-libs,ALL -static-libstdc++ -static-libgcc)
endif()

# OFX bundle layout: MoGe3.ofx.bundle/Contents/<arch>/MoGe3.ofx
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/MoGe3.ofx.bundle/Contents/${OFX_ARCH_DIR}")
set_target_properties(MoGe3 PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  RUNTIME_OUTPUT_DIRECTORY "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BUNDLE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${BUNDLE_DIR}")

# ---- test host (not part of the bundle) ------------------------------------
option(MOGE3_BUILD_TESTS "build the mini OFX host used by ofx/tests" ON)
if(MOGE3_BUILD_TESTS)
  add_executable(mini_host tests/mini_host.cpp)
  target_include_directories(mini_host PRIVATE include/openfx)
  if(NOT WIN32)
    target_link_libraries(mini_host PRIVATE dl)
  endif()
  set_target_properties(mini_host PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
endif()
