include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(optonaut-lib
common/image.cpp
common/progressCallback.cpp
common/static_timer.cpp
io/checkpointStore.cpp
io/inputImage.cpp
io/io.cpp
math/quat.cpp
math/support.cpp
recorder/recorder.cpp
stereo/monoStitcher.cpp
stitcher/dynamicSeamer.cpp
stitcher/multiringStitcher.cpp
stitcher/ringStitcher.cpp
stitcher/simpleSphereStitcher.cpp
stitcher/simplePlaneStitcher.cpp
)
target_link_libraries(optonaut-lib ${OpenCV_LIBS})

add_library(visual-debug-hook
debug/visualDebugHook.cpp
)
target_link_libraries(visual-debug-hook ${OpenCV_LIBS})
target_link_libraries(visual-debug-hook optonaut-lib)
target_link_libraries(visual-debug-hook ${OPENGL_LIBRARIES})
target_link_libraries(visual-debug-hook ${EXTRA_LIBS})
target_link_libraries(visual-debug-hook ${CMAKE_CURRENT_LIST_DIR}/../vendor/libIrrlicht.a)

add_executable(to-polar toPol.cpp)
target_link_libraries(to-polar optonaut-lib)

add_executable(optonaut-test stitcherTest.cpp)
target_link_libraries(optonaut-test optonaut-lib)

add_executable(aligner-test alignerTest.cpp)
target_link_libraries(aligner-test optonaut-lib)

add_executable(perspective-aligner-test perspectiveAlignerTest.cpp)
target_link_libraries(perspective-aligner-test optonaut-lib)

add_executable(ring-closure-test ringClosureTest.cpp)
target_link_libraries(ring-closure-test optonaut-lib)

add_executable(debug-hook-test debugHookTest.cpp)
target_link_libraries(debug-hook-test visual-debug-hook)

add_executable(stereo-conversion-test stereoConversionTest.cpp)
target_link_libraries(stereo-conversion-test optonaut-lib)

add_subdirectory(test)
add_subdirectory(debug)
