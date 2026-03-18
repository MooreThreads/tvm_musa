if(USE_MUSA)
    if(DEFINED ENV{MUSA_HOME})
        set(MUSA_HOME $ENV{MUSA_HOME})
    else()
        set(MUSA_HOME "/usr/local/musa")
    endif()
    list(APPEND CMAKE_MODULE_PATH ${MUSA_HOME}/cmake)
    find_package(MUSAToolkit REQUIRED)
    if(NOT MUSAToolkit_FOUND)
        message(FATAL_ERROR "Cannot find MUSA, USE_MUSA=" ${USE_MUSA})
    endif()
    message(STATUS "Build with MUSA ${MUSAToolkit_VERSION} support")

    message(STATUS "MUSA INCLUDE DIR: ${MUSAToolkit_INCLUDE_DIRS}")
    message(STATUS "MUSA LIBRARY DIR: ${MUSAToolkit_LIBRARY_DIR}")
    message(STATUS "MUSA TARGET DIR: ${MUSAToolkit_TARGET_DIR}")

    tvm_file_glob(GLOB RUNTIME_MUSA_SRCS src/runtime/musa/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_MUSA_SRCS})
    list(APPEND COMPILER_SRCS src/target/opt/build_musa_on.cc)

    set_property(SOURCE
        src/target/opt/build_musa_on.cc
        ${RUNTIME_MUSA_SRCS}
        APPEND PROPERTY INCLUDE_DIRECTORIES ${MUSAToolkit_INCLUDE_DIRS})

    list(APPEND TVM_RUNTIME_LINKER_LIBS MUSA::musart)
    list(APPEND TVM_RUNTIME_LINKER_LIBS MUSA::musa_driver)

else(USE_MUSA)
    list(APPEND COMPILER_SRCS src/target/opt/build_musa_off.cc)
endif(USE_MUSA)
