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
 * \file musa_device_api.cc
 * \brief GPU specific API
 */
#include <musa.h>
#include <musa_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/timer.h>

#include <cstring>

#include "musa_common.h"

namespace tvm {
namespace runtime {

class MUSADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { MUSA_CALL(musaSetDevice(dev.device_id)); }
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    int value = 0;
    switch (kind) {
      case kExist: {
        int count;
        auto err = musaGetDeviceCount(&count);
        value = (err == musaSuccess && static_cast<int>(dev.device_id < count));
        break;
      }
      case kMaxThreadsPerBlock: {
        MUSA_CALL(musaDeviceGetAttribute(&value, musaDevAttrMaxThreadsPerBlock, dev.device_id));
        break;
      }
      case kWarpSize: {
        MUSA_CALL(musaDeviceGetAttribute(&value, musaDevAttrWarpSize, dev.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        MUSA_CALL(
            musaDeviceGetAttribute(&value, musaDevAttrMaxSharedMemoryPerBlock, dev.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        MUSA_CALL(musaDeviceGetAttribute(&value, musaDevAttrComputeCapabilityMajor, dev.device_id));
        os << value << ".";
        MUSA_CALL(musaDeviceGetAttribute(&value, musaDevAttrComputeCapabilityMinor, dev.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        std::string name(256, 0);
        MUSA_DRIVER_CALL(muDeviceGetName(&name[0], name.size(), dev.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        MUSA_CALL(musaDeviceGetAttribute(&value, musaDevAttrClockRate, dev.device_id));
        break;
      }
      case kMultiProcessorCount: {
        MUSA_CALL(musaDeviceGetAttribute(&value, musaDevAttrMultiProcessorCount, dev.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        MUSA_CALL(musaDeviceGetAttribute(&dims[0], musaDevAttrMaxBlockDimX, dev.device_id));
        MUSA_CALL(musaDeviceGetAttribute(&dims[1], musaDevAttrMaxBlockDimY, dev.device_id));
        MUSA_CALL(musaDeviceGetAttribute(&dims[2], musaDevAttrMaxBlockDimZ, dev.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
      case kMaxRegistersPerBlock: {
        MUSA_CALL(musaDeviceGetAttribute(&value, musaDevAttrMaxRegistersPerBlock, dev.device_id));
        break;
      }
      case kGcnArch:
        return;
      case kApiVersion: {
        *rv = MUSA_VERSION;
        return;
      }
      case kDriverVersion:
        return;
      case kL2CacheSizeBytes: {
        // Get size of device l2 cache size in bytes.
        int l2_size = 0;
        MUSA_CALL(musaDeviceGetAttribute(&l2_size, musaDevAttrL2CacheSize, dev.device_id));
        *rv = l2_size;
        return;
      }
      case kTotalGlobalMemory: {
        musaDeviceProp prop;
        MUSA_CALL(musaGetDeviceProperties(&prop, dev.device_id));
        int64_t total_global_memory = prop.totalGlobalMem;
        *rv = total_global_memory;
        return;
      }
      case kAvailableGlobalMemory: {
        size_t free_mem, total_mem;
        MUSA_CALL(musaMemGetInfo(&free_mem, &total_mem));
        *rv = static_cast<int64_t>(free_mem);
        return;
      }
      case kImagePitchAlignment:
        return;
    }
    *rv = value;
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    TVM_FFI_ICHECK_EQ(256 % alignment, 0U) << "MUSA space is aligned at 256 bytes";
    void* ret;
    MUSA_CALL(musaSetDevice(dev.device_id));
    size_t free_mem, total_mem;
    MUSA_CALL(musaMemGetInfo(&free_mem, &total_mem));
    VLOG(1) << "allocating " << nbytes << " bytes on device, with " << free_mem
            << " bytes currently free out of " << total_mem << " bytes available";
    MUSA_CALL(musaMalloc(&ret, nbytes));
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    if (std::uncaught_exceptions() && musaPeekAtLastError() == musaErrorIllegalAddress) {
      // For most MUSA calls, an error from an API call will be
      // immediately reported, and raised as an exception.  However,
      // errors raised from async kernel execution leave the MUSA
      // driver in an inconsistent state.  These errors are "sticky",
      // and are never cleared. (See [0] for more details.)
      //
      // If we are currently unwinding the stack due to a thrown
      // exception, and the MUSA driver is in an unrecoverable error,
      // do not attempt to free the MUSA allocations.  Performing any
      // MUSA API call while in this state will throw an additional
      // exception, causing a segfault.  In this case, it is better to
      // allow the original error to continue propagating.
      //
      // [0] https://forums.developer.nvidia.com/t/musa-errors-determine-sticky-ness/271625
      return;
    }


    MUSA_CALL(musaSetDevice(dev.device_id));
    VLOG(1) << "freeing device memory";
    MUSA_CALL(musaFree(ptr));
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    musaStream_t cu_stream = static_cast<musaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    // In case there is a copy from host mem to host mem */
    if (dev_to.device_type == kDLCPU && dev_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (dev_from.device_type == kDLExtDev && dev_to.device_type == kDLExtDev) {
      MUSA_CALL(musaSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, musaMemcpyDeviceToDevice, cu_stream);
      } else {
        musaMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, cu_stream);
      }
    } else if (dev_from.device_type == kDLExtDev && dev_to.device_type == kDLCPU) {
      MUSA_CALL(musaSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, musaMemcpyDeviceToHost, cu_stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLExtDev) {
      MUSA_CALL(musaSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, musaMemcpyHostToDevice, cu_stream);
    } else {
      TVM_FFI_THROW(InternalError) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) {
    MUSA_CALL(musaSetDevice(dev.device_id));
    musaStream_t retval;
    MUSA_CALL(musaStreamCreateWithFlags(&retval, musaStreamNonBlocking));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    MUSA_CALL(musaSetDevice(dev.device_id));
    musaStream_t cu_stream = static_cast<musaStream_t>(stream);
    MUSA_CALL(musaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    MUSA_CALL(musaSetDevice(dev.device_id));
    musaStream_t src_stream = static_cast<musaStream_t>(event_src);
    musaStream_t dst_stream = static_cast<musaStream_t>(event_dst);
    musaEvent_t evt;
    MUSA_CALL(musaEventCreate(&evt));
    MUSA_CALL(musaEventRecord(evt, src_stream));
    MUSA_CALL(musaStreamWaitEvent(dst_stream, evt, 0));
    MUSA_CALL(musaEventDestroy(evt));
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    MUSA_CALL(musaSetDevice(dev.device_id));
    MUSA_CALL(musaStreamSynchronize(static_cast<musaStream_t>(stream)));
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return MUSAThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
  }

  void FreeWorkspace(Device dev, void* data) final {
    MUSAThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
  }

  bool SupportsDevicePointerArithmeticsOnHost() final { return true; }

  static MUSADeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new MUSADeviceAPI();
    return inst;
  }

 private:
  static void GPUCopy(const void* from, void* to, size_t size, musaMemcpyKind kind,
                      musaStream_t stream) {
    MUSA_CALL(musaMemcpyAsync(to, from, size, kind, stream));
  }
};

MUSAThreadEntry::MUSAThreadEntry() : pool(kDLExtDev, MUSADeviceAPI::Global()) {}

MUSAThreadEntry* MUSAThreadEntry::ThreadLocal() {
  static thread_local MUSAThreadEntry inst;
  return &inst;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.musa",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    DeviceAPI* ptr = MUSADeviceAPI::Global();
                    *rv = static_cast<void*>(ptr);
                  })
      .def_packed("device_api.musa_host", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = MUSADeviceAPI::Global();
        *rv = static_cast<void*>(ptr);
      });
}

class MUSATimerNode : public TimerNode {
 public:
  virtual void Start() {
    // This initial musaEventRecord is sometimes pretty slow (~100us). Does
    // musaEventRecord do some stream synchronization?
    int device_id;
    MUSA_CALL(musaGetDevice(&device_id));
    stream_ = TVMFFIEnvGetStream(kDLExtDev, device_id);
    MUSA_CALL(musaEventRecord(start_, static_cast<musaStream_t>(stream_)));
  }
  virtual void Stop() {
    int device_id;
    MUSA_CALL(musaGetDevice(&device_id));
    MUSA_CALL(musaEventRecord(stop_, static_cast<musaStream_t>(stream_)));
  }
  virtual int64_t SyncAndGetElapsedNanos() {
    MUSA_CALL(musaEventSynchronize(stop_));
    float milliseconds = 0;
    MUSA_CALL(musaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~MUSATimerNode() {
    MUSA_CALL(musaEventDestroy(start_));
    MUSA_CALL(musaEventDestroy(stop_));
  }
  MUSATimerNode() {
    MUSA_CALL(musaEventCreate(&start_));
    MUSA_CALL(musaEventCreate(&stop_));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.musa.MUSATimerNode", MUSATimerNode, TimerNode);

 private:
  musaEvent_t start_;
  musaEvent_t stop_;
  TVMStreamHandle stream_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.musa",
                        [](Device dev) { return Timer(ffi::make_object<MUSATimerNode>()); });
}

TVM_RUNTIME_DLL ffi::String GetMusaFreeMemory() {
  size_t free_mem, total_mem;
  MUSA_CALL(musaMemGetInfo(&free_mem, &total_mem));
  std::stringstream ss;
  ss << "Current MUSA memory is " << free_mem << " bytes free out of " << total_mem
     << " bytes on device";
  return ss.str();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.GetMusaFreeMemory", GetMusaFreeMemory)
      .def("runtime.get_musa_stream", []() {
        // TODO(tvm-team): remove once confirms all dep such as flashinfer
        // migrated to TVMFFIEnvGetStream
        int device_id;
        MUSA_CALL(musaGetDevice(&device_id));
        return static_cast<void*>(TVMFFIEnvGetStream(kDLExtDev, device_id));
      });
}

TVM_RUNTIME_DLL int GetMusaDeviceCount() {
  int count;
  MUSA_CALL(musaGetDeviceCount(&count));
  return count;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.GetMusaDeviceCount", GetMusaDeviceCount);
}


}  // namespace runtime
}  // namespace tvm
