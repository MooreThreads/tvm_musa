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
 * \file musa_module.cc
 */
#include "musa_module.h"

#include <musa.h>
#include <musa_runtime.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "../../support/bytes_io.h"
#include "../file_utils.h"
#include "../metadata.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "musa_common.h"

namespace tvm {
namespace runtime {

namespace {

constexpr const char* kMUSAGlobalBarrierState = "__tvm_global_barrier_state";
constexpr const char* kMUSAPrepareGlobalBarrier = "__tvm_prepare_global_barrier";

std::atomic<bool>& MUSAProcessShutdownFlag() {
  // Leak the flag intentionally so it remains valid even during late process
  // teardown when module destructors may still run.
  static auto* flag = new std::atomic<bool>(false);
  return *flag;
}

}  // namespace

// Module to support thread-safe multi-GPU execution.
// muModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class MUSAModuleNode : public ffi::ModuleObj {
 public:
  explicit MUSAModuleNode(ffi::Bytes data, ffi::String fmt,
                          ffi::Map<ffi::String, FunctionInfo> fmap,
                          ffi::Map<ffi::String, ffi::String> source)
      : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~MUSAModuleNode() {
    if (MUSAProcessShutdownFlag().load(std::memory_order_relaxed)) {
      return;
    }
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        MUSA_CALL(musaSetDevice(static_cast<int>(i)));
        MUSA_DRIVER_CALL(muModuleUnload(module_[i]));
      }
    }
  }

  const char* kind() const final { return "musa"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "mu") {
      auto it = source_.find("musa");
      TVM_FFI_ICHECK(it != source_.end());
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, std::string((*it).second));
    } else {
      TVM_FFI_ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, std::string(data_.data(), data_.size()));
    }
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(data_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) {
      return ffi::String(data_.data(), data_.size());
    }
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      if (auto it = source_.find("musa"); it != source_.end()) {
        return (*it).second;
      }
    }
    if (fmt_ == "ptx") {
      return ffi::String(data_.data(), data_.size());
    }
    return "";
  }

  // get a MUfunction from primary context in device_id
  MUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      MUSA_DRIVER_CALL(muModuleLoadData(&(module_[device_id]), data_.data()));
      static auto nvshmem_init_hook = ffi::Function::GetGlobal("runtime.nvshmem.mumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void*>(module_[device_id]));
      }
    }
    MUfunction func;
    MUresult result = muModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != MUSA_SUCCESS) {
      const char* msg;
      muGetErrorName(result, &msg);
      LOG(FATAL) << "MUSAError: muModuleGetFunction " << func_name << " failed with error: " << msg;
    }
    return func;
  }
  // get a global var from primary context in device_id
  MUdeviceptr GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      MUSA_DRIVER_CALL(muModuleLoadData(&(module_[device_id]), data_.data()));
      static auto nvshmem_init_hook = ffi::Function::GetGlobal("runtime.nvshmem.mumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void*>(module_[device_id]));
      }
    }
    MUdeviceptr global;
    size_t nbytes;

    MUresult result = muModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str());
    TVM_FFI_ICHECK_EQ(nbytes, expect_nbytes);
    if (result != MUSA_SUCCESS) {
      const char* msg;
      muGetErrorName(result, &msg);
      LOG(FATAL) << "MUSAError: muModuleGetGlobal " << global_name << " failed with error: " << msg;
    }
    return global;
  }

 private:
  // the binary data
  ffi::Bytes data_;
  // The format
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // Optional source map for inspection.
  ffi::Map<ffi::String, ffi::String> source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<MUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class MUSAWrappedFunc {
 public:
  // initialize the MUSA function.
  void Init(MUSAModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr, const std::string& func_name,
            size_t num_void_args, const ffi::Array<ffi::String>& launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    // Track whether this kernel uses dynamic shared memory and the last size set per device.
    std::fill(dyn_smem_initialized_.begin(), dyn_smem_initialized_.end(), false);
    // Track whether cluster attribute has been set per device.
    std::fill(cluster_attr_initialized_.begin(), cluster_attr_initialized_.end(), false);
    use_dyn_shared_memory_ = false;
    for (const auto& tag : launch_param_tags) {
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        use_dyn_shared_memory_ = true;
        break;
      }
    }
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
    int device_id;
    MUSA_CALL(musaGetDevice(&device_id));
    ThreadWorkLoad wl = launch_param_config_.Extract(args);

    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    // If the kernel uses dynamic shared memory, we should ensure the attribute
    // reflects the actual size needed for this launch. Some workloads vary the
    // dynamic shared memory between invocations, in which case we cannot set it
    // just once. Cache the last value per device to avoid redundant calls.
    bool need_dyn_attr = use_dyn_shared_memory_ || (wl.dyn_shmem_size > 0);
    if (need_dyn_attr) {
      if (!dyn_smem_initialized_[device_id] || dyn_smem_last_[device_id] != wl.dyn_shmem_size) {
        MUresult attr_set = muFuncSetAttribute(
            fcache_[device_id], MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, wl.dyn_shmem_size);
        if (attr_set != MUSA_SUCCESS) {
          LOG(FATAL) << "Failed to set the allowed dynamic shared memory size to "
                     << wl.dyn_shmem_size;
        }
        dyn_smem_last_[device_id] = wl.dyn_shmem_size;
        dyn_smem_initialized_[device_id] = true;
      }
    }
    MUstream strm = static_cast<MUstream>(TVMFFIEnvGetStream(kDLExtDev, device_id));
    MUresult result;

    if (wl.use_cluster_launch()) {
      // SM90+ cluster launch
      MUlaunchConfig config{};
      MUlaunchAttribute attribute[2]{};
      attribute[0].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attribute[0].value.clusterDim.x = wl.cluster_dim[0];
      attribute[0].value.clusterDim.y = wl.cluster_dim[1];
      attribute[0].value.clusterDim.z = wl.cluster_dim[2];
      attribute[1].id = MU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[1].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 2;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      // Set non-portable cluster size allowed attribute
      if (!cluster_attr_initialized_[device_id]) {
        MUresult attr_result = muFuncSetAttribute(
            fcache_[device_id], MU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
        if (attr_result != MUSA_SUCCESS) {
          const char* msg;
          muGetErrorName(attr_result, &msg);
          LOG(FATAL) << "Failed to set cluster attribute for " << func_name_ << ": " << msg;
        }
        cluster_attr_initialized_[device_id] = true;
      }

      result = muLaunchKernelEx(&config, fcache_[device_id], void_args, nullptr);
    } else if (launch_param_config_.use_programtic_dependent_launch()) {
      MUlaunchConfig config{};
      MUlaunchAttribute attribute[1]{};
      attribute[0].id = MU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[0].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 1;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      result = muLaunchKernelEx(&config, fcache_[device_id], void_args, nullptr);
    } else if (launch_param_config_.use_cooperative_launch()) {
      result = static_cast<MUresult>(
          musaLaunchCooperativeKernel(reinterpret_cast<const void*>(fcache_[device_id]),
                                      dim3(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2)),
                                      dim3(wl.block_dim(0), wl.block_dim(1), wl.block_dim(2)),
                                      void_args, wl.dyn_shmem_size, strm));
    } else {
      result = muLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
                              wl.block_dim(0), wl.block_dim(1), wl.block_dim(2), wl.dyn_shmem_size,
                              strm, void_args, nullptr);
    }

    if (result != MUSA_SUCCESS && result != MUSA_ERROR_DEINITIALIZED) {
      const char* msg = nullptr;
      muGetErrorName(result, &msg);
      std::ostringstream os;
      os << "MUSALaunch Error: " << (msg ? msg : "unknown")
         << " (code=" << static_cast<int>(result) << ")\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2)
         << ")"
         << " dyn_smem_bytes=" << wl.dyn_shmem_size;
      auto append_func_attr = [&](MUfunction_attribute attr, const char* name) {
        int value = 0;
        MUresult attr_result = muFuncGetAttribute(&value, attr, fcache_[device_id]);
        os << "\n " << name << "=";
        if (attr_result == MUSA_SUCCESS) {
          os << value;
        } else {
          os << "unavailable(code=" << static_cast<int>(attr_result) << ")";
        }
      };
      append_func_attr(MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "max_threads_per_block");
      append_func_attr(MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, "static_smem_bytes");
      append_func_attr(MU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, "const_bytes");
      append_func_attr(MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, "local_bytes");
      append_func_attr(MU_FUNC_ATTRIBUTE_NUM_REGS, "num_regs");
      append_func_attr(MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, "max_dynamic_smem_bytes");
      std::string musa = m_->InspectSource("");
      if (musa.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// MUSA Source\n"
           << "// -----------\n"
           << musa;
      }
      LOG(FATAL) << os.str();
    }

    // Check for asynchronous MUSA errors that muLaunchKernel's return value
    // does not capture (e.g. illegal memory access during kernel execution).
    // This mirrors the CUDA FFI backend and Cython backend's
    // TILELANG_CHECK_LAST_ERROR behavior.
    if (result == MUSA_SUCCESS) {
      musaError_t last_err = musaPeekAtLastError();
      if (last_err != musaSuccess) {
        const char* err_name = musaGetErrorName(last_err);
        const char* err_str = musaGetErrorString(last_err);
        // Clear the sticky error so subsequent MUSA calls are not poisoned.
        musaGetLastError();
        LOG(FATAL) << func_name_ << ": " << (err_name ? err_name : "unknown") << " - "
                   << err_str;
      }
    }
  }

 private:
  // internal module
  MUSAModuleNode* m_;
  // the resource holder
  ffi::ObjectPtr<ffi::Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<MUfunction, kMaxNumGPUs> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
  // Whether this kernel uses dynamic shared memory
  bool use_dyn_shared_memory_{false};
  // Cached last dynamic shared memory size per device and whether it's initialized
  mutable std::array<size_t, kMaxNumGPUs> dyn_smem_last_;
  mutable std::array<bool, kMaxNumGPUs> dyn_smem_initialized_;
  // Whether cluster attribute has been initialized per device
  mutable std::array<bool, kMaxNumGPUs> cluster_attr_initialized_;
  // have pdl setting
  bool has_programmatic_dependent_launch_;
};

class MUSAPrepGlobalBarrier {
 public:
  MUSAPrepGlobalBarrier(MUSAModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr) : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  void operator()(const ffi::PackedArgs& args, ffi::Any* rv) const {
    int device_id;
    MUSA_CALL(musaGetDevice(&device_id));
    if (pcache_[device_id] == 0) {
      pcache_[device_id] =
          m_->GetGlobal(device_id, kMUSAGlobalBarrierState, sizeof(unsigned));
    }
    MUSA_DRIVER_CALL(muMemsetD32(pcache_[device_id], 0, 1));
  }

 private:
  // internal module
  MUSAModuleNode* m_;
  // the resource holder
  ffi::ObjectPtr<ffi::Object> sptr_;
  // mark as mutable, to enable lazy initialization
  mutable std::array<MUdeviceptr, kMaxNumGPUs> pcache_;
};

ffi::Optional<ffi::Function> MUSAModuleNode::GetFunction(const ffi::String& name) {
  ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  if (name == kMUSAPrepareGlobalBarrier) {
    return ffi::Function(MUSAPrepGlobalBarrier(this, sptr_to_self));
  }
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return ffi::Function();
  FunctionInfo info = opt_info.value();
  MUSAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info->arg_types.size(), info->launch_param_tags);
  return PackFuncVoidAddr(f, info->arg_types, info->arg_extra_tags);
}

ffi::Module MUSAModuleCreate(std::string data, std::string fmt,
                             ffi::Map<ffi::String, FunctionInfo> fmap, std::string musa_source) {
  ffi::Map<ffi::String, ffi::String> source;
  if (!musa_source.empty()) {
    source.Set("musa", ffi::String(musa_source));
  }
  auto n = ffi::make_object<MUSAModuleNode>(ffi::Bytes(std::move(data)), ffi::String(fmt), fmap,
                                           source);
  return ffi::Module(n);
}

// Load module from module.
ffi::Module MUSAModuleLoadFile(const std::string& file_name, const ffi::String& format) {
  std::string data;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return MUSAModuleCreate(data, fmt, fmap, std::string());
}

ffi::Module MUSAModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes data;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&data);
  auto n = ffi::make_object<MUSAModuleNode>(std::move(data), std::move(fmt), std::move(fmap),
                                           ffi::Map<ffi::String, ffi::String>());
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.musa.mark_shutdown", []() {
        MUSAProcessShutdownFlag().store(true, std::memory_order_relaxed);
      })
      .def("ffi.Module.load_from_file.musa", MUSAModuleLoadFile)
      .def("ffi.Module.load_from_bytes.musa", MUSAModuleLoadFromBytes);
}
}  // namespace runtime
}  // namespace tvm
