add_library(osx-support
    threadDetach.mm)

target_link_libraries(osx-support ${OSX_LIBS})
