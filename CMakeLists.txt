
cmake_minimum_required(VERSION 2.8)
project(optonaut-stitcher)
include(GNUInstallDirs)

#find_package(OpenGL REQUIRED)
#include_directories(${OPENGL_INCLUDE_DIR})
#include_directories(SYSTEM src/lib/irrlicht)
#include_directories("/usr/local/include/Eigen3")

INCLUDE_DIRECTORIES ( /System/Library/Frameworks )

#FIND_LIBRARY( COCOA_LIB Cocoa )
#FIND_LIBRARY( CARBON_LIB Carbon )
#FIND_LIBRARY( IOKIT_LIB IOKit )

#MARK_AS_ADVANCED (COCOA_LIB
#    CARBON_LIB 
#    IOKIT_LIB)
#SET(OSX_LIBS ${COCOA_LIB} ${CARBON_LIB} ${IOKIT_LIB})

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wunreachable-code -std=c++1y -O3 -g -Werror")
add_subdirectory(src)
