cmake_minimum_required(VERSION 3.16)
project(tcapi_itensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- ITensor (external dependency, pre-built by its own Make workflow) ----
# Headers are included as <itensor/all.h>, resolved against <src>/itensor.
# Library: itensor/lib/libitensor.a (release build). Rebuild with `make -C itensor`.
set(ITENSOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/itensor)
set(ITENSOR_LIB ${ITENSOR_DIR}/lib/libitensor.a)

if(NOT EXISTS ${ITENSOR_LIB})
  message(FATAL_ERROR
    "ITensor library not found at ${ITENSOR_LIB}.\n"
    "ITensor must be built first with its own Make workflow:  make -C ${ITENSOR_DIR}")
endif()

add_library(itensor STATIC IMPORTED GLOBAL)
set_target_properties(itensor PROPERTIES
  IMPORTED_LOCATION ${ITENSOR_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${ITENSOR_DIR})

if(APPLE)
  find_library(ACCELERATE_LIBRARY Accelerate)
  if(NOT ACCELERATE_LIBRARY)
    message(FATAL_ERROR "Accelerate framework not found (required to link ITensor on macOS).")
  endif()
  target_link_libraries(itensor INTERFACE ${ACCELERATE_LIBRARY})
endif()

# ---- TCAPI (header-only interface library) ----
add_library(tcapi INTERFACE)
target_include_directories(tcapi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tcapi INTERFACE itensor)
target_compile_options(tcapi INTERFACE
  -Wall -Wno-unknown-pragmas -Wno-unused-variable)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(tcapi INTERFACE -Wno-gcc-compat)
endif()

# ---- Tests ----
enable_testing()

set(TEST_SOURCES
  tests/test_constructors.cc
  tests/test_contract.cc
  tests/test_errors.cc
  tests/test_examples.cc
  tests/test_io_misc.cc
  tests/test_linalg.cc
  tests/test_manipulation.cc)

foreach(src ${TEST_SOURCES})
  get_filename_component(tname ${src} NAME_WE)
  add_executable(${tname} ${src})
  target_link_libraries(${tname} PRIVATE tcapi)
  add_test(NAME ${tname} COMMAND ${tname})
endforeach()

# ---- Example ----
add_executable(trg examples/trg.cc)
target_link_libraries(trg PRIVATE tcapi)

add_executable(itebd examples/itebd.cc)
target_link_libraries(itebd PRIVATE tcapi)
