# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.5.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.5.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mcandres/sandbox/optonaut/app-ios/stitcher

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build

# Include any dependencies generated for this target.
include src/CMakeFiles/to-polar.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/to-polar.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/to-polar.dir/flags.make

src/CMakeFiles/to-polar.dir/toPol.cpp.o: src/CMakeFiles/to-polar.dir/flags.make
src/CMakeFiles/to-polar.dir/toPol.cpp.o: ../src/toPol.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/to-polar.dir/toPol.cpp.o"
	cd /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/to-polar.dir/toPol.cpp.o -c /Users/mcandres/sandbox/optonaut/app-ios/stitcher/src/toPol.cpp

src/CMakeFiles/to-polar.dir/toPol.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/to-polar.dir/toPol.cpp.i"
	cd /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mcandres/sandbox/optonaut/app-ios/stitcher/src/toPol.cpp > CMakeFiles/to-polar.dir/toPol.cpp.i

src/CMakeFiles/to-polar.dir/toPol.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/to-polar.dir/toPol.cpp.s"
	cd /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mcandres/sandbox/optonaut/app-ios/stitcher/src/toPol.cpp -o CMakeFiles/to-polar.dir/toPol.cpp.s

src/CMakeFiles/to-polar.dir/toPol.cpp.o.requires:

.PHONY : src/CMakeFiles/to-polar.dir/toPol.cpp.o.requires

src/CMakeFiles/to-polar.dir/toPol.cpp.o.provides: src/CMakeFiles/to-polar.dir/toPol.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/to-polar.dir/build.make src/CMakeFiles/to-polar.dir/toPol.cpp.o.provides.build
.PHONY : src/CMakeFiles/to-polar.dir/toPol.cpp.o.provides

src/CMakeFiles/to-polar.dir/toPol.cpp.o.provides.build: src/CMakeFiles/to-polar.dir/toPol.cpp.o


# Object files for target to-polar
to__polar_OBJECTS = \
"CMakeFiles/to-polar.dir/toPol.cpp.o"

# External object files for target to-polar
to__polar_EXTERNAL_OBJECTS =

src/to-polar: src/CMakeFiles/to-polar.dir/toPol.cpp.o
src/to-polar: src/CMakeFiles/to-polar.dir/build.make
src/to-polar: src/liboptonaut-lib.a
src/to-polar: /usr/local/lib/libopencv_xphoto.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_ximgproc.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_tracking.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_surface_matching.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_saliency.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_rgbd.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_reg.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_optflow.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_line_descriptor.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_latentsvm.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_datasets.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_text.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_face.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_ccalib.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_bioinspired.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_bgsegm.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_adas.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_xobjdetect.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_videostab.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_superres.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_stitching.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_xfeatures2d.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_shape.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_video.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_photo.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_objdetect.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_calib3d.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_features2d.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_ml.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_highgui.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_videoio.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_imgcodecs.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_imgproc.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_flann.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_core.3.0.0.dylib
src/to-polar: /usr/local/lib/libopencv_hal.a
src/to-polar: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
src/to-polar: src/CMakeFiles/to-polar.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable to-polar"
	cd /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/to-polar.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/to-polar.dir/build: src/to-polar

.PHONY : src/CMakeFiles/to-polar.dir/build

src/CMakeFiles/to-polar.dir/requires: src/CMakeFiles/to-polar.dir/toPol.cpp.o.requires

.PHONY : src/CMakeFiles/to-polar.dir/requires

src/CMakeFiles/to-polar.dir/clean:
	cd /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/src && $(CMAKE_COMMAND) -P CMakeFiles/to-polar.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/to-polar.dir/clean

src/CMakeFiles/to-polar.dir/depend:
	cd /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mcandres/sandbox/optonaut/app-ios/stitcher /Users/mcandres/sandbox/optonaut/app-ios/stitcher/src /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/src /Users/mcandres/sandbox/optonaut/app-ios/stitcher/build/src/CMakeFiles/to-polar.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/to-polar.dir/depend
