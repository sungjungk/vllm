#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
//
// When `require_live_view` is true the function will raise instead of falling
// through to the detached alloc+copy path. Callers that depend on
// write-through coherence (e.g. V2 UVA buffers) must set this flag.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor, bool require_live_view) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = cpu_tensor.layout();
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  // CUDA can classify CC pinned allocations as Managed, for which
  // aten::is_pinned is false. With HMM, mapping can also succeed for ordinary
  // unregistered storage, which must retain the detached fallback contract.
  void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
  cudaPointerAttributes attributes{};
  cudaError_t err = cudaPointerGetAttributes(&attributes, host_ptr);
  STD_TORCH_CHECK(err == cudaSuccess || err == cudaErrorInvalidValue,
                  "cudaPointerGetAttributes failed with unexpected error: ",
                  cudaGetErrorString(err));
  const bool registered =
      err == cudaSuccess && attributes.type != cudaMemoryTypeUnregistered;
  void* device_ptr = nullptr;
  if (registered) {
    err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  }
  if (registered && err == cudaSuccess) {
    return torch::stable::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_dev, dtype,
        [base = cpu_tensor](void*) {});  // keep cpu tensor alive
  }

  STD_TORCH_CHECK(err == cudaSuccess || err == cudaErrorInvalidValue,
                  "cudaHostGetDevicePointer failed with unexpected error: ",
                  cudaGetErrorString(err));
  // Clear a non-fatal attribute/mapping error before throwing or falling back.
  if (err != cudaSuccess) cudaGetLastError();

  STD_TORCH_CHECK(!require_live_view,
                  "get_cuda_view_from_cpu_tensor: host memory is not "
                  "registered for zero-copy access but require_live_view=true. "
                  "The returned view would be a detached copy without "
                  "write-through coherence.");

  // Preserve the compatibility fallback for callers that do not require a
  // live alias. Subsequent writes to cpu_tensor are not visible through it.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* new_host_ptr = nullptr;
  err = cudaHostAlloc(&new_host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(new_host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(new_host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, new_host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(new_host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [new_host_ptr](void*) { cudaFreeHost(new_host_ptr); };

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type(), deleter);
}
