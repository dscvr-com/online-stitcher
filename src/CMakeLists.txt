
add_library(optonaut-lib
common/image.cpp
common/progressCallback.cpp
common/static_timer.cpp
common/static_counter.cpp
common/jniHelper.cpp
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
stitcher/flowBlender.cpp
stitcher/simpleSphereStitcher.cpp
stitcher/simplePlaneStitcher.cpp
debug/debugHook.cpp
)
target_link_libraries(optonaut-lib ${OpenCV_LIBS})

#add_library(visual-debug-hook
#debug/visualDebugHook.cpp
#)

#target_link_libraries(visual-debug-hook ${OpenCV_LIBS})
#target_link_libraries(visual-debug-hook optonaut-lib)
#target_link_libraries(visual-debug-hook ${OPENGL_LIBRARIES})
#target_link_libraries(visual-debug-hook ${OSX_LIBS})
#target_link_libraries(visual-debug-hook ${CMAKE_CURRENT_LIST_DIR}/../vendor/libIrrlicht.a)

if(BUILD_STABILIZATION_TEST) 
    add_executable(stabilization-test stabilizationTest.cpp)
    target_link_libraries(stabilization-test optonaut-lib)
endif(BUILD_STABILIZATION_TEST) 

if(BUILD_SIMPLE_STITCHER_TEST) 
    add_executable(simple-stitcher simpleStitcherTest.cpp)
    target_link_libraries(simple-stitcher optonaut-lib)
endif(BUILD_SIMPLE_STITCHER_TEST) 

if(BUILD_TO_POLAR) 
    add_executable(to-polar toPol.cpp)
    target_link_libraries(to-polar optonaut-lib)
endif(BUILD_TO_POLAR) 

if(BUILD_TO_CUBE_MAP) 
    add_executable(to-cube-map toCubeMap.cpp)
    target_link_libraries(to-cube-map optonaut-lib)
endif(BUILD_TO_CUBE_MAP) 

if(BUILD_PANO_BLUR)
    add_executable(pano-blur panoBlur.cpp)
    target_link_libraries(pano-blur optonaut-lib)
endif(BUILD_PANO_BLUR)

if(BUILD_STITCHER_TEST)
    add_executable(optonaut-test stitcherTest.cpp)
    target_link_libraries(optonaut-test optonaut-lib)
endif(BUILD_STITCHER_TEST)

if(BUILD_ALIGNER_TEST)
    add_executable(aligner-test alignerTest.cpp)
    target_link_libraries(aligner-test optonaut-lib)
endif(BUILD_ALIGNER_TEST)

if(BUILD_PERSPECTIVE_ALIGNER_TEST)
    add_executable(perspective-aligner-test perspectiveAlignerTest.cpp)
    target_link_libraries(perspective-aligner-test optonaut-lib)
endif(BUILD_PERSPECTIVE_ALIGNER_TEST)

if(BUILD_RING_CLOSURE_TEST)
    add_executable(ring-closure-test ringClosureTest.cpp)
    target_link_libraries(ring-closure-test optonaut-lib)
endif(BUILD_RING_CLOSURE_TEST)

#add_executable(debug-hook-test debugHookTest.cpp)
#target_link_libraries(debug-hook-test visual-debug-hook)

if(BUILD_STEREO_CONVERSION_TEST)
    add_executable(stereo-conversion-test stereoConversionTest.cpp)
    target_link_libraries(stereo-conversion-test optonaut-lib)
endif(BUILD_STEREO_CONVERSION_TEST)

#add_executable(feature-chain-test featureChainTest.cpp)
#target_link_libraries(feature-chain-test optonaut-lib)

#add_executable(minimal-feature-test minimalFeatureTest.cpp)
#target_link_libraries(minimal-feature-test optonaut-lib)

#add_executable(sfm-test sfmTest.cpp)
#target_link_libraries(sfm-test optonaut-lib)
#target_link_libraries(sfm-test visual-debug-hook)

if(BUILD_BRUTE_FORCE_ALIGNER_TEST)
    add_executable(brute-force-aligner-test bruteForceAlignerTest.cpp)
    target_link_libraries(brute-force-aligner-test optonaut-lib)
endif(BUILD_BRUTE_FORCE_ALIGNER_TEST)

if(BUILD_FLOW_FIELD_TEST)
    add_executable(flow-field-test flowFieldLKTest.cpp)
    target_link_libraries(flow-field-test optonaut-lib)
endif(BUILD_FLOW_FIELD_TEST)

if(BUILD_FLOW_BLENDING_TEST)
    add_executable(flow-blending-test flowBlendingTest.cpp)
    target_link_libraries(flow-blending-test optonaut-lib)
endif(BUILD_FLOW_BLENDING_TEST)

if(BUILD_SPEED_TEST)
    add_executable(speed-test speedTest.cpp)
    target_link_libraries(speed-test optonaut-lib)
endif(BUILD_SPEED_TEST)

if(BUILD_DENSE_FLOW_TEST)
    add_executable(flow-test flowTest.cpp)
    target_link_libraries(flow-test optonaut-lib)
endif(BUILD_DENSE_FLOW_TEST)

if(BUILD_SFML_TEST)
    add_executable(sfml-test sfmlTest.cpp)
    target_link_libraries(sfml-test ${SFML_LIBRARIES})
    target_link_libraries(sfml-test ${OPENGL_LIBRARIES})
    target_link_libraries(sfml-test optonaut-lib)
    target_link_libraries(sfml-test ${OSX_LIBS})
endif(BUILD_SFML_TEST)

#add_executable(stereo-match-test stereoMatchTest.cpp)
#target_link_libraries(stereo-match-test optonaut-lib)
#target_link_libraries(stereo-match-test visual-debug-hook)

add_subdirectory(test)
#add_subdirectory(debug)
