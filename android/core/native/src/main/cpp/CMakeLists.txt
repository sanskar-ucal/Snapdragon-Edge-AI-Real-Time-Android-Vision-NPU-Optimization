cmake_minimum_required(VERSION 3.22)
project(edgevision_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(preprocess_core STATIC preprocess.cpp)
target_include_directories(preprocess_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(preprocess_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ANDROID)
  add_library(edgevision_native SHARED jni_bridge.cpp)
  target_link_libraries(edgevision_native PRIVATE preprocess_core jnigraphics log)
  # 16 KB page alignment required for Android 15+ devices.
  target_link_options(edgevision_native PRIVATE "-Wl,-z,max-page-size=16384")
else()
  # Host build: `cmake -S . -B build && cmake --build build && ctest --test-dir build`
  enable_testing()
  add_executable(preprocess_test test/preprocess_test.cpp)
  target_link_libraries(preprocess_test PRIVATE preprocess_core)
  add_test(NAME preprocess_test COMMAND preprocess_test)
endif()
