include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(optonaut-test quat.cpp io.cpp image.cpp support.cpp monoStitcher.cpp simpleSphereStitcher.cpp testbed.cpp pipeline.cpp ringwiseStitcher.cpp static_timer.cpp inputImage.cpp)

target_link_libraries( optonaut-test ${OpenCV_LIBS})
