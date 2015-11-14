add_executable(stat-test statTest.cpp)
target_link_libraries(stat-test ${OpenCV_LIBS})
target_link_libraries(stat-test optonaut-lib)

add_executable(processor-test processorTest.cpp)
target_link_libraries(processor-test ${OpenCV_LIBS})
target_link_libraries(processor-test optonaut-lib)



