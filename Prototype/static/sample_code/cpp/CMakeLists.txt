cmake_minimum_required(VERSION 3.15)
project(ep2c_mock)
add_executable(ep2c src/main.cpp src/model.cpp)
target_include_directories(ep2c PRIVATE include)
