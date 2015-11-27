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

set_target_properties(optonaut-lib PROPERTIES COMPILE_FLAGS "-Werror")

add_executable(irrlicht-test irrlichtTest.cpp)
target_link_libraries(irrlicht-test ${CMAKE_CURRENT_LIST_DIR}/../vendor/libIrrlicht.a)
target_link_libraries(irrlicht-test ${OPENGL_LIBRARIES})
target_link_libraries(irrlicht-test ${EXTRA_LIBS})

add_executable(to-polar toPol.cpp)
target_link_libraries(to-polar ${OpenCV_LIBS})

add_executable(optonaut-test stitcherTest.cpp)
target_link_libraries(optonaut-test ${OpenCV_LIBS})
target_link_libraries(optonaut-test optonaut-lib)

add_executable(aligner-test alignerTest.cpp)
target_link_libraries(aligner-test ${OpenCV_LIBS})
target_link_libraries(aligner-test optonaut-lib)

add_executable(perspective-aligner-test perspectiveAlignerTest.cpp)
target_link_libraries(perspective-aligner-test ${OpenCV_LIBS})
target_link_libraries(perspective-aligner-test optonaut-lib)

add_executable(ring-closure-test ringClosureTest.cpp)
target_link_libraries(ring-closure-test ${OpenCV_LIBS})
target_link_libraries(ring-closure-test optonaut-lib)

add_executable(stereo-conversion-test stereoConversionTest.cpp)
target_link_libraries(stereo-conversion-test ${OpenCV_LIBS})
target_link_libraries(stereo-conversion-test optonaut-lib)

add_subdirectory(test)
