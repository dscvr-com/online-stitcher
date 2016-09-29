add_executable(stat-test statTest.cpp)
target_link_libraries(stat-test optonaut-lib)

add_executable(quat-test quatTest.cpp)
target_link_libraries(quat-test optonaut-lib)

add_executable(processor-test processorTest.cpp)
target_link_libraries(processor-test optonaut-lib)

add_executable(async-queue-test asyncQueueTest.cpp)
#target_link_libraries(async-queue-test optonaut-lib)

add_executable(slerp-test slerpTest.cpp)
target_link_libraries(slerp-test optonaut-lib)

add_executable(graph-test graphTest.cpp)
target_link_libraries(graph-test optonaut-lib)

