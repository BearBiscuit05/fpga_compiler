cmake_minimum_required(VERSION 3.10)

project(Test)

if(NOT DEFINED ENV{INTELFPGAOCLSDKROOT})
    message(FATAL_ERROR "not define intel FPGA SDK environment...")
endif()

find_package(OpenCL)
message("OpenCL is find : " ${OpenCL_FOUND})
message("PATH " ${OpenCL_LIBRARIES})

