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
#include <dmlc/memory_io.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "musa_common.h"

namespace tvm {
namespace runtime {

namespace {

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
  explicit MUSAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string musa_source)
      : data_(data), fmt_(fmt), fmap_(fmap), musa_source_(musa_source) {
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
      ICHECK_NE(musa_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, musa_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    dmlc::MemoryStringStream ms(&buffer);
    dmlc::Stream* stream = &ms;
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
    return ffi::Bytes(buffer);
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) return data_;
    if (musa_source_.length() != 0) {
      return musa_source_;
    } else {
      if (fmt_ == "ptx") return data_;
      return "";
    }
  }

  // get a MUfunction from primary context in device_id
  MUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      MUSA_DRIVER_CALL(muModuleLoadData(&(module_[device_id]), data_.c_str()));
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
      MUSA_DRIVER_CALL(muModuleLoadData(&(module_[device_id]), data_.c_str()));
      static auto nvshmem_init_hook = ffi::Function::GetGlobal("runtime.nvshmem.mumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void*>(module_[device_id]));
      }
    }
    MUdeviceptr global;
    size_t nbytes;

    MUresult result = muModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str());
    ICHECK_EQ(nbytes, expect_nbytes);
    if (result != MUSA_SUCCESS) {
      const char* msg;
      muGetErrorName(result, &msg);
      LOG(FATAL) << "MUSAError: muModuleGetGlobal " << global_name << " failed with error: " << msg;
    }
    return global;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The musa source.
  std::string musa_source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<MUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class MUSAWrappedFunc {
 public:
  // initialize the MUSA function.
  void Init(MUSAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& launch_param_tags) {
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
      const char* msg;
      muGetErrorName(result, &msg);
      std::ostringstream os;
      os << "MUSALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2)
         << ")"
         << " dyn_smem_bytes=" << wl.dyn_shmem_size;
      std::string musa = m_->InspectSource("");
      if (musa.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// MUSA Source\n"
           << "// -----------\n"
           << musa;
      }
      LOG(FATAL) << os.str();
    }
  }

 private:
  // internal module
  MUSAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
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
  MUSAPrepGlobalBarrier(MUSAModuleNode* m, ObjectPtr<Object> sptr) : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  void operator()(const ffi::PackedArgs& args, ffi::Any* rv) const {
    int device_id;
    MUSA_CALL(musaGetDevice(&device_id));
    if (pcache_[device_id] == 0) {
      pcache_[device_id] =
          m_->GetGlobal(device_id, runtime::symbol::tvm_global_barrier_state, sizeof(unsigned));
    }
    MUSA_DRIVER_CALL(muMemsetD32(pcache_[device_id], 0, 1));
  }

 private:
  // internal module
  MUSAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // mark as mutable, to enable lazy initialization
  mutable std::array<MUdeviceptr, kMaxNumGPUs> pcache_;
};

ffi::Optional<ffi::Function> MUSAModuleNode::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);
  ICHECK_EQ(sptr_to_self.get(), this);
  if (name == symbol::tvm_prepare_global_barrier) {
    return ffi::Function(MUSAPrepGlobalBarrier(this, sptr_to_self));
  }
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return ffi::Function();
  const FunctionInfo& info = it->second;
  MUSAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types, info.arg_extra_tags);
}

ffi::Module MUSAModuleCreate(std::string data, std::string fmt,
                             std::unordered_map<std::string, FunctionInfo> fmap,
                             std::string musa_source) {
  auto n = ffi::make_object<MUSAModuleNode>(data, fmt, fmap, musa_source);
  return ffi::Module(n);
}

// Load module from module.
ffi::Module MUSAModuleLoadFile(const std::string& file_name, const ffi::String& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return MUSAModuleCreate(data, fmt, fmap, std::string());
}

ffi::Module MUSAModuleLoadFromBytes(const ffi::Bytes& bytes) {
  dmlc::MemoryFixedSizeStream ms(const_cast<char*>(bytes.data()), bytes.size());
  dmlc::Stream* stream = &ms;
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return MUSAModuleCreate(data, fmt, fmap, std::string());
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
