cmake_minimum_required(VERSION 3.22)
project(MolDy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -pthread)

include_directories(.)

# add_executable(MolDy
 #       altpotential.cpp
  #      BS_thread_pool.hpp
   #     celllists.cpp
 #       multithread.cpp
  #      refactor.cpp
   #     twoatoms.cpp)
#
add_executable(multithread multithread.cpp BS_thread_pool.hpp)
add_executable(arrayofvectors arrayofvectors.cpp)
add_executable(working working.cpp)
add_executable(noloops noloops.cpp)