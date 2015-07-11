include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(lib/tinyxml2)

add_executable(optonaut-test quat.cpp core.cpp support.cpp monoStitcher.cpp simpleSphereStitcher.cpp testbed.cpp)
target_link_libraries( optonaut-test ${OpenCV_LIBS} tinyxml2)
