set(ProjName facemask)
project(${ProjName})
cmake_minimum_required(VERSION 2.8.11)

AUX_SOURCE_DIRECTORY(. SRC_LIST)
add_executable(${ProjName} ${SRC_LIST})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_DIRS})
target_link_libraries (${ProjName} ${OpenCV_LIBS})