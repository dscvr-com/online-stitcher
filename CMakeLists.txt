
cmake_minimum_required(VERSION 2.8)
project(optonaut-stitcher)
include(GNUInstallDirs)

option(BUILD_SIMPLE_STITCHER_TEST "Build simple stitching program" ON) 
option(BUILD_TO_POLAR "Build to polar conversion tool" ON) 
option(BUILD_TO_CUBE_MAP "Build equirectangular to cube map conversion tool" ON) 
option(BUILD_PANO_BLUR "Build panorama blur tool" ON) 
option(BUILD_STITCHER_TEST "Build main optonaut pipeline test" ON) 
option(BUILD_ALIGNER_TEST "Build aligner test" ON) 
option(BUILD_PERSPECTIVE_ALIGNER_TEST "Build perspective aligner test" ON) 
option(BUILD_RING_CLOSURE_TEST "Build ring closure test" ON) 
option(BUILD_STEREO_CONVERSION_TEST "Build stereo conversion test" ON) 
option(BUILD_BRUTE_FORCE_ALIGNER_TEST "Build iterative bindle aligner test" ON) 

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wunreachable-code -std=c++1y -O0 -g -Werror")
add_subdirectory(src)
