#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> slice_acquisition_forward_cuda(
    torch::Tensor transforms,
    torch::Tensor vol,
    torch::Tensor vol_mask,
    torch::Tensor slices_mask,
    torch::Tensor psf,
    torch::IntArrayRef slice_shape,
    const float res_slice,
    const bool need_weight,
    const bool interp_psf);

std::vector<torch::Tensor> slice_acquisition_backward_cuda(
    torch::Tensor transforms,
    torch::Tensor vol,
    torch::Tensor vol_mask,
    torch::Tensor psf,
    torch::Tensor grad_slices,
    torch::Tensor slices_mask,
    const float res_slice,
    const bool interp_psf,
    const bool need_vol_grad,
    const bool need_transforms_grad);

std::vector<torch::Tensor> slice_acquisition_adjoint_forward_cuda(
    torch::Tensor transforms,
    torch::Tensor psf,
    torch::Tensor slices,
    torch::Tensor slices_mask,
    torch::Tensor vol_mask,
    torch::IntArrayRef vol_shape,
    const float res_slice,
    const bool interp_psf,
    const bool equalize);

std::vector<torch::Tensor> slice_acquisition_adjoint_backward_cuda(
    torch::Tensor transforms,
    torch::Tensor grad_vol,
    torch::Tensor vol_weight,
    torch::Tensor vol_mask,
    torch::Tensor psf,
    torch::Tensor slices,
    torch::Tensor slices_mask,
    torch::Tensor vol,
    const float res_slice,
    const bool interp_psf,
    const bool equalize,
    const bool need_slices_grad,
    const bool need_transforms_grad);
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> slice_acquisition_forward(
    torch::Tensor transforms,
    torch::Tensor vol,
    torch::Tensor vol_mask,
    torch::Tensor slices_mask,
    torch::Tensor psf,
    torch::IntArrayRef slice_shape,
    const float res_slice,
    const bool need_weight,
    const bool interp_psf) {
  CHECK_INPUT(transforms);
  CHECK_INPUT(vol);
  CHECK_INPUT(psf);
  CHECK_INPUT(vol_mask);
  CHECK_INPUT(slices_mask);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(transforms));
  return slice_acquisition_forward_cuda(transforms, vol, vol_mask, slices_mask, psf, slice_shape, res_slice, need_weight, interp_psf);
}

std::vector<torch::Tensor> slice_acquisition_backward(
    torch::Tensor transforms,
    torch::Tensor vol,
    torch::Tensor vol_mask,
    torch::Tensor psf,
    torch::Tensor grad_slices,
    torch::Tensor slices_mask,
    const float res_slice,
    const bool interp_psf,
    const bool need_vol_grad,
    const bool need_transforms_grad) {
  CHECK_INPUT(transforms);
  CHECK_INPUT(vol);
  CHECK_INPUT(psf);
  CHECK_INPUT(grad_slices);
  CHECK_INPUT(vol_mask);
  CHECK_INPUT(slices_mask);
  

  const at::cuda::OptionalCUDAGuard device_guard(device_of(transforms));
  return slice_acquisition_backward_cuda(transforms, vol, vol_mask, psf, grad_slices, slices_mask, res_slice, interp_psf, need_vol_grad, need_transforms_grad);
}


std::vector<torch::Tensor> slice_acquisition_adjoint_forward(
    torch::Tensor transforms,
    torch::Tensor psf,
    torch::Tensor slices,
    torch::Tensor slices_mask,
    torch::Tensor vol_mask,
    torch::IntArrayRef vol_shape,
    const float res_slice,
    const bool interp_psf,
    const bool equalize) {
  CHECK_INPUT(transforms);
  CHECK_INPUT(psf);
  CHECK_INPUT(slices);
  CHECK_INPUT(slices_mask);
  CHECK_INPUT(vol_mask);
  

  const at::cuda::OptionalCUDAGuard device_guard(device_of(transforms));
  return slice_acquisition_adjoint_forward_cuda(transforms, psf, slices, slices_mask, vol_mask, vol_shape, res_slice, interp_psf, equalize);
}

std::vector<torch::Tensor> slice_acquisition_adjoint_backward(
    torch::Tensor transforms,
    torch::Tensor grad_vol,
    torch::Tensor vol_weight,
    torch::Tensor vol_mask,
    torch::Tensor psf,
    torch::Tensor slices,
    torch::Tensor slices_mask,
    torch::Tensor vol,
    const float res_slice,
    const bool interp_psf,
    const bool equalize,
    const bool need_slices_grad,
    const bool need_transforms_grad) {
  CHECK_INPUT(transforms);
  CHECK_INPUT(psf);
  CHECK_INPUT(slices);
  CHECK_INPUT(grad_vol);
  CHECK_INPUT(vol_mask);
  CHECK_INPUT(slices_mask);
  if (equalize){
    CHECK_INPUT(vol);
    CHECK_INPUT(vol_weight);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(transforms));
  return slice_acquisition_adjoint_backward_cuda(transforms, grad_vol, vol_weight, vol_mask, psf, slices, slices_mask, vol, res_slice, interp_psf, equalize, need_slices_grad, need_transforms_grad);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &slice_acquisition_forward, "slice acquisition forward (CUDA)");
  m.def("backward", &slice_acquisition_backward, "slice acquisition backward (CUDA)");
  m.def("adjoint_forward", &slice_acquisition_adjoint_forward, "slice acquisition adjoint forward (CUDA)");
  m.def("adjoint_backward", &slice_acquisition_adjoint_backward, "slice acquisition adjoint backward (CUDA)");
}