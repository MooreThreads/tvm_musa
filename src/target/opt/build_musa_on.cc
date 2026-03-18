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
 *  Build cuda modules from source.
 *  requires cuda to be available.
 *
 * \file build_cuda.cc
 */
#include "tvm/ffi/extra/module.h"
#if defined(__linux__)
#include <sys/stat.h>
#include <tvm/ffi/reflection/registry.h>
#endif
// #include <musa_runtime.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../support/process_id.h"
#include "../../support/utils.h"
#include "../../runtime/file_utils.h"
#include "../../runtime/musa/musa_common.h"
#include "../../runtime/musa/musa_module.h"
#include "../build_common.h"
#include "../source/codegen_musa.h"
#include <tvm/ffi/string.h>


namespace tvm {
namespace codegen {

std::string MTCompile(const std::string& code, const Target& target) {
  static std::atomic<uint64_t> counter{0};

  std::filesystem::path base_dir(runtime::GetCacheDir());
  std::error_code dir_err;
  std::filesystem::create_directories(base_dir, dir_err);
  ICHECK(!dir_err) << "MTCompile: cannot create cache directory " << base_dir.string() << ": "
                   << dir_err.message();

  std::ostringstream name;
  name << "musa_kernel_" << support::GetProcessId() << "_" << counter.fetch_add(1);

  const std::string source_path = (base_dir / (name.str() + ".mu")).string();
  const std::string output_path = (base_dir / (name.str() + ".mubin")).string();

  {
    std::ofstream ofs(source_path, std::ios::out | std::ios::trunc);
    ICHECK(ofs.good()) << "MTCompile: cannot open temp source file " << source_path;
    ofs << code;
    ofs.close();
    ICHECK(!ofs.fail()) << "MTCompile: failed to flush source file " << source_path;
  }

  const char* compiler_env = std::getenv("TVM_MUSA_COMPILER");
  std::string compiler = compiler_env != nullptr ? compiler_env : "mcc";

  auto arch = target->GetAttr<tvm::ffi::String>("arch");
  if (!arch.has_value()) {
    LOG(FATAL) << "target does not have arch attribute";
  }

  std::vector<std::string> args;
  args.push_back("--cuda-device-only");
  args.push_back("--cuda-gpu-arch=" + arch.value());

  if (const char* extra_env = std::getenv("TVM_MUSA_FLAGS")) {
    std::istringstream flags_stream(extra_env);
    std::string flag;
    while (flags_stream >> flag) {
      args.push_back(flag);
    }
  }

  args.push_back("-o");
  args.push_back(output_path);
  args.push_back(source_path);

  std::ostringstream cmd_builder;
  cmd_builder << compiler;
  for (const auto& arg : args) {
    cmd_builder << " " << arg;
  }

  std::string err_msg;
  int ret_code = support::Execute(cmd_builder.str(), &err_msg);
  ICHECK_EQ(ret_code, 0) << "MTCompile failed (code " << ret_code << ")\n"
                         << "Command: " << cmd_builder.str() << "\n"
                         << err_msg;

  std::string compiled;
  runtime::LoadBinaryFromFile(output_path, &compiled);

  std::cout << "MTCompile: " << source_path << " " << output_path << std::endl;

  // runtime::RemoveFile(source_path);
  // runtime::RemoveFile(output_path);



  return compiled;
}

ffi::Module BuildMUSA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenMUSA cg;
  cg.Init(output_ssa);

  ffi::Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenMUSA: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    auto calling_conv =
        prim_func->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(tvm::CallingConv::kDefault));
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch ||
           calling_conv == CallingConv::kDefault)
        << "CodeGenMUSA: expect calling_conv equals CallingConv::kDeviceKernelLaunch or "
           "CallingConv::kDefault";
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  std::string code = cg.Finish();

  if (auto f = ffi::Function::GetGlobal("tvm_callback_musa_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  std::string mubin = MTCompile(code, target);

  return MUSAModuleCreate(mubin, "mubin", ExtractFuncInfo(mod), code);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.musa", BuildMUSA);
}
TVM_REGISTER_PASS_CONFIG_OPTION("musa.kernels_output_dir", ffi::String);
}  // namespace codegen
}  // namespace tvm
