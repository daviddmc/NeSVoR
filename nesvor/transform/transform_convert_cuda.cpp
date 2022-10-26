#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> axisangle2mat_forward_cuda(
    torch::Tensor axisangle);

std::vector<torch::Tensor> axisangle2mat_backward_cuda(
    torch::Tensor grad_mat,
    torch::Tensor axisangle);

std::vector<torch::Tensor> mat2axisangle_forward_cuda(
    torch::Tensor mat);

std::vector<torch::Tensor> mat2axisangle_backward_cuda(
    torch::Tensor mat,
    torch::Tensor grad_axisangle);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> axisangle2mat_forward(
    torch::Tensor axisangle) {
  CHECK_INPUT(axisangle);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(axisangle));
  return axisangle2mat_forward_cuda(axisangle);
}

std::vector<torch::Tensor> axisangle2mat_backward(
    torch::Tensor grad_mat,
    torch::Tensor axisangle) {
  CHECK_INPUT(axisangle);
  CHECK_INPUT(grad_mat);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(axisangle));
  return axisangle2mat_backward_cuda(grad_mat, axisangle);
}

std::vector<torch::Tensor> mat2axisangle_forward(
    torch::Tensor mat) {
  CHECK_INPUT(mat);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return mat2axisangle_forward_cuda(mat);
}

std::vector<torch::Tensor> mat2axisangle_backward(
    torch::Tensor mat,
    torch::Tensor grad_axisangle) {
  CHECK_INPUT(grad_axisangle);
  CHECK_INPUT(mat);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return mat2axisangle_backward_cuda(mat, grad_axisangle);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axisangle2mat_forward", &axisangle2mat_forward, "axisangle2mat forward (CUDA)");
  m.def("axisangle2mat_backward", &axisangle2mat_backward, "axisangle2mat backward (CUDA)");
  m.def("mat2axisangle_forward", &mat2axisangle_forward, "mat2axisangle forward (CUDA)");
  m.def("mat2axisangle_backward", &mat2axisangle_backward, "mat2axisangle backward (CUDA)");
}