# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# MUSA Module
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

  set_property(SOURCE
    ${RUNTIME_MUSA_SRCS}
    APPEND PROPERTY INCLUDE_DIRECTORIES ${MUSAToolkit_INCLUDE_DIRS})

  list(APPEND TVM_RUNTIME_LINKER_LIBS MUSA::musart)
  list(APPEND TVM_RUNTIME_LINKER_LIBS MUSA::musa_driver)
endif(USE_MUSA)
