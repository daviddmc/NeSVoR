from argparse import Namespace
from typing import List
import torch
from ..transform import transform_points
from ..image import Slice, Volume
from .models import INR
from ..utils import resolution2sigma, meshgrid


def sample_volume(model: INR, mask: Volume, args: Namespace) -> Volume:
    model.eval()
    img = mask.resample(args.output_resolution, None)
    img.image[img.mask] = sample_points(model, img.xyz_masked, args)
    return img


def sample_points(model: INR, xyz: torch.Tensor, args: Namespace) -> torch.Tensor:
    shape = xyz.shape[:-1]
    xyz = xyz.view(-1, 3)
    v = torch.empty(xyz.shape[0], dtype=torch.float32, device=args.device)
    batch_size = args.inference_batch_size
    with torch.no_grad():
        for i in range(0, xyz.shape[0], batch_size):
            xyz_batch = xyz[i : i + batch_size]
            xyz_batch = model.sample_batch(
                xyz_batch,
                None,
                resolution2sigma(args.output_resolution, isotropic=True),
                0 if args.no_output_psf else args.n_inference_samples,
            )
            v_b = model(xyz_batch, False).mean(-1)
            v[i : i + batch_size] = v_b
    return v.view(shape)


def sample_slice(model: INR, slice: Slice, mask: Volume, args: Namespace) -> Slice:
    # clone the slice
    slice_sampled = slice.clone()
    slice_sampled.image = torch.zeros_like(slice_sampled.image)
    slice_sampled.mask = torch.zeros_like(slice_sampled.mask)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0
    if m.any():
        xyz_masked = model.sample_batch(
            xyz[m],
            slice_sampled.transformation,
            resolution2sigma(slice_sampled.resolution_xyz, isotropic=False),
            0 if args.no_output_psf else args.n_inference_samples,
        )
        v = model(xyz_masked, False).mean(-1)
        slice_sampled.mask = m.view(slice_sampled.mask.shape)
        slice_sampled.image[slice_sampled.mask] = v.to(slice_sampled.image.dtype)
    return slice_sampled


def sample_slices(
    model: INR, slices: List[Slice], mask: Volume, args: Namespace
) -> List[Slice]:
    model.eval()
    with torch.no_grad():
        slices_sampled = []
        for i, slice in enumerate(slices):
            slices_sampled.append(sample_slice(model, slice, mask, args))
    return slices_sampled
