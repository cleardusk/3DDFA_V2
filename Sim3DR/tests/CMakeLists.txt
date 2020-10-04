cmake_minimum_required(VERSION 2.8)

set(TARGET test)
project(${TARGET})

#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -O3")
include_directories(../lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")
add_executable(${TARGET} test.cpp ../lib/rasterize_kernel.cpp io.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
