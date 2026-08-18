/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file musa_common.h
 * \brief Common utilities for MUSA
 */
#ifndef TVM_RUNTIME_MUSA_MUSA_COMMON_H_
#define TVM_RUNTIME_MUSA_MUSA_COMMON_H_

#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <string>

#include "../workspace_pool.h"

namespace tvm {
namespace runtime {

#define MUSA_DRIVER_CALL(x)                                             \
  {                                                                     \
    MUresult result = x;                                                \
    if (result != MUSA_SUCCESS && result != MUSA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      muGetErrorName(result, &msg);                                     \
      TVM_FFI_THROW(InternalError) << "MUSAError: " #x " failed with error: " << msg;     \
    }                                                                   \
  }

#define MUSA_CALL(func)                                       \
  {                                                           \
    musaError_t e = (func);                                   \
    TVM_FFI_ICHECK(e == musaSuccess || e == musaErrorMusartUnloading) \
        << "MUSA: " << musaGetErrorString(e);                 \
  }

/*! \brief Thread local workspace */
class MUSAThreadEntry {
 public:
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  MUSAThreadEntry();
  // get the threadlocal workspace
  static MUSAThreadEntry* ThreadLocal();
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MUSA_MUSA_COMMON_H_
