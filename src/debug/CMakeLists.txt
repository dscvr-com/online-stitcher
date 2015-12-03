add_executable(irrlicht-test irrlichtTest.cpp)
target_link_libraries(irrlicht-test ${CMAKE_CURRENT_LIST_DIR}/../../vendor/libIrrlicht.a)
target_link_libraries(irrlicht-test ${OPENGL_LIBRARIES})
target_link_libraries(irrlicht-test ${EXTRA_LIBS})

add_executable(irrlicht-threading-test irrlichtThreadingTest.cpp)
target_link_libraries(irrlicht-threading-test ${CMAKE_CURRENT_LIST_DIR}/../../vendor/libIrrlicht.a)
target_link_libraries(irrlicht-threading-test ${OPENGL_LIBRARIES})
target_link_libraries(irrlicht-threading-test ${EXTRA_LIBS})

add_custom_command(TARGET irrlicht-test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_LIST_DIR}/assets $<TARGET_FILE_DIR:irrlicht-test>)
