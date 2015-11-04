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
)

add_executable(optonaut-test stitcherTest.cpp)
target_link_libraries(optonaut-test ${OpenCV_LIBS})
target_link_libraries(optonaut-test optonaut-lib)
