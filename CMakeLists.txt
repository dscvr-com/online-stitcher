
cmake_minimum_required(VERSION 2.8)
project(optonaut-stitcher)
include(GNUInstallDirs)

option(BUILD_STABILIZATION_TEST "Build stabilization test program" ON) 
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
option(BUILD_FLOW_FIELD_TEST "Build flow field test program" OFF) 
option(BUILD_FLOW_BLENDING_TEST "Build flow field blending test program" OFF)
option(BUILD_DENSE_FLOW_TEST "Build flow algorithm tests" OFF) 
option(BUILD_SPEED_TEST "Build implementation detail speed tests" ON) 
option(BUILD_SFML_TEST "Build sfml GLSL processing test" ON) 

#include_directories(SYSTEM src/lib/irrlicht)
#include_directories("/usr/local/include/Eigen3")

#INCLUDE_DIRECTORIES ( /System/Library/Frameworks )

FIND_LIBRARY( COCOA_LIB Cocoa )
FIND_LIBRARY( CARBON_LIB Carbon )
FIND_LIBRARY( IOKIT_LIB IOKit )

MARK_AS_ADVANCED (COCOA_LIB
    CARBON_LIB 
    IOKIT_LIB)
SET(OSX_LIBS ${COCOA_LIB} ${CARBON_LIB} ${IOKIT_LIB})

# Add support for custon cmake include scripts
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

# Add modules
find_package( OpenCV REQUIRED )
find_package( OpenGL REQUIRED )
find_package( SFML 2 REQUIRED system graphics window )
find_package( OpenGL REQUIRED )
include_directories($ {OPENGL_INCLUDE_DIR} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${OPENGL_INCLUDE_DIR} )
include_directories( ${SFML_INCLUDE_DIR} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wunreachable-code -std=c++1y -O3 -g -fverbose-asm")
add_subdirectory(src)
