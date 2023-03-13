// native_specialization_constant() returns true only in JIT mode
// on opencl & level-zero backends (because only SPIR-V supports them)

// REQUIRES: opencl, level-zero, cpu, gpu, opencl-aot, ocloc

// RUN: %clangxx -DJIT -fsycl %s -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &h) { h.single_task<>([]() {}); });

#ifdef JIT
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Q.get_context());
  assert(bundle.native_specialization_constant());
#else
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context());
  // This assert will fail in JIT mode, because there are no images in
  // executable state, so native_specialization_constant() will return true
  assert(!bundle.native_specialization_constant());
#endif // JIT

  return 0;
}
