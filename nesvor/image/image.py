from __future__ import annotations
import os
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from ..transform import RigidTransform, transform_points
from .image_utils import (
    affine2transformation,
    compare_resolution_affine,
    transformation2affine,
)
from ..utils import meshgrid


class Image(object):
    def __init__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: Union[float, torch.Tensor] = 1.0,
        resolution_y: Union[float, torch.Tensor] = 1.0,
        resolution_z: Union[float, torch.Tensor] = 1.0,
    ) -> None:
        assert image.ndim == 3
        self.image = image
        if mask is None:
            mask = torch.ones_like(image, dtype=torch.bool)
        self.mask = mask
        if transformation is None:
            transformation = RigidTransform(
                torch.zeros((1, 6), dtype=torch.float32, device=image.device)
            )
        self.transformation = transformation
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

    def clone(self, zero: bool = False):
        raise NotImplementedError

    def _clone_image(self, zero: bool = False) -> Dict:
        return {
            "image": torch.zeros_like(self.image) if zero else self.image.clone(),
            "mask": torch.zeros_like(self.mask) if zero else self.mask.clone(),
            "transformation": self.transformation.clone(),
            "resolution_x": float(self.resolution_x),
            "resolution_y": float(self.resolution_y),
            "resolution_z": float(self.resolution_z),
        }

    @property
    def shape_xyz(self) -> torch.Tensor:
        return torch.tensor(self.image.shape[::-1], device=self.image.device)

    @property
    def resolution_xyz(self) -> torch.Tensor:
        return torch.tensor(
            [self.resolution_x, self.resolution_y, self.resolution_z],
            device=self.image.device,
        )

    def save(self, path: str, masked=True) -> None:
        affine = transformation2affine(
            self.image,
            self.transformation,
            float(self.resolution_x),
            float(self.resolution_y),
            float(self.resolution_z),
        )
        if masked:
            output_volume = self.image * self.mask.to(self.image.dtype)
        else:
            output_volume = self.image
        save_nii_volume(path, output_volume, affine)

    @property
    def xyz_masked(self) -> torch.Tensor:
        return transform_points(self.transformation, self.xyz_masked_untransformed)

    @property
    def xyz_masked_untransformed(self) -> torch.Tensor:
        kji = torch.flip(torch.nonzero(self.mask), (-1,))
        return (kji - (self.shape_xyz - 1) / 2) * self.resolution_xyz

    @property
    def v_masked(self) -> torch.Tensor:
        return self.image[self.mask]

    def rescale(self, intensity_mean: Union[float, torch.Tensor]) -> None:
        scale_factor = intensity_mean / self.image[self.mask].mean()
        self.image *= scale_factor


class Slice(Image):
    def __init__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: Union[float, torch.Tensor] = 1.0,
        resolution_y: Union[float, torch.Tensor] = 1.0,
        resolution_z: Union[float, torch.Tensor] = 1.0,
        stack_idx: Optional[int] = None,
        slice_idx: Optional[int] = None,
    ) -> None:
        super().__init__(
            image, mask, transformation, resolution_x, resolution_y, resolution_z
        )
        self.stack_idx = stack_idx
        self.slice_idx = slice_idx

    def clone(self, zero: bool = False) -> Slice:
        return Slice(
            stack_idx=self.stack_idx,
            slice_idx=self.slice_idx,
            **self._clone_image(zero),
        )


class Volume(Image):
    def sample_points(self, xyz: torch.Tensor) -> torch.Tensor:
        shape = xyz.shape[:-1]
        xyz = transform_points(self.transformation.inv(), xyz.view(-1, 3))
        xyz = xyz / ((self.shape_xyz - 1) * self.resolution_xyz / 2)
        return F.grid_sample(
            self.image[None, None],
            xyz.view(1, 1, 1, -1, 3),
            align_corners=True,
        ).view(shape)

    def resample(
        self,
        resolution_new: Optional[Union[float, torch.Tensor]],
        transformation_new: Optional[RigidTransform],
    ) -> Volume:

        if transformation_new is None:
            transformation_new = self.transformation
        R = transformation_new.matrix()[0, :3, :3]
        dtype = R.dtype
        device = R.device
        if resolution_new is None:
            resolution_new = self.resolution_xyz
        elif isinstance(resolution_new, float) or resolution_new.numel == 1:
            resolution_new = torch.tensor(
                [resolution_new] * 3, dtype=dtype, device=device
            )

        xyz = self.xyz_masked
        # new rotation
        xyz = torch.matmul(torch.inverse(R), xyz.view(-1, 3, 1))[..., 0]

        xyz_min = xyz.amin(0) - resolution_new * 10
        xyz_max = xyz.amax(0) + resolution_new * 10
        shape_xyz = ((xyz_max - xyz_min) / resolution_new).ceil().long()

        mat = torch.zeros((1, 3, 4), dtype=R.dtype, device=R.device)
        mat[0, :, :3] = R
        mat[0, :, -1] = xyz_min + (shape_xyz - 1) / 2 * resolution_new

        xyz = meshgrid(shape_xyz, resolution_new, xyz_min, device, True)

        xyz = torch.matmul(R, xyz[..., None])[..., 0]

        v = self.sample_points(xyz)

        return Volume(
            v,
            v > 0,
            RigidTransform(mat, trans_first=True),
            resolution_new[0].item(),
            resolution_new[1].item(),
            resolution_new[2].item(),
        )

    def clone(self, zero: bool = False) -> Volume:
        return Volume(**self._clone_image(zero))


class Stack(object):
    def __init__(
        self,
        slices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        score: float = 0.0,
        resolution_x: float = 1.0,
        resolution_y: float = 1.0,
        thickness: float = 1.0,
        gap: float = 1.0,
    ) -> None:
        self.slices = slices
        if mask is None:
            # mask = slices > 0.0
            mask = torch.ones_like(slices, dtype=torch.bool)
        self.mask = mask
        if transformation is None:
            t = torch.zeros(
                (slices.shape[0], 6), dtype=torch.float32, device=slices.device
            )
            t[:, -1] = (
                torch.arange(slices.shape[0], dtype=torch.float32, device=slices.device)
                - slices.shape[0] / 2
            ) * gap
            transformation = RigidTransform(t)
        self.transformation = transformation
        if score is None:
            score = torch.ones(
                slices.shape[0], dtype=torch.float32, device=slices.device
            )
        self.score = score
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.thickness = thickness
        self.gap = gap

    def __len__(self) -> int:
        return self.slices.shape[0]

    def __getitem__(self, idx):
        assert self.slices.ndim == 4
        slices = self.slices[idx]
        masks = self.mask[idx]
        transformation = self.transformation[idx]
        if slices.ndim < self.slices.ndim:
            return Slice(
                slices,
                masks,
                transformation,
                self.resolution_x,
                self.resolution_y,
                self.thickness,
            )
        else:
            return [
                Slice(
                    slices[i],
                    masks[i],
                    transformation[i],
                    self.resolution_x,
                    self.resolution_y,
                    self.thickness,
                )
                for i in range(len(transformation))
            ]


def save_nii_volume(
    path: str,
    volume: Union[torch.Tensor, np.ndarray],
    affine: Optional[Union[torch.Tensor, np.ndarray]],
) -> None:
    assert len(volume.shape) == 3 or (len(volume.shape) == 4 and volume.shape[1] == 1)
    if len(volume.shape) == 4:
        volume = volume.squeeze(1)
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy().transpose(2, 1, 0)
    else:
        volume = volume.transpose(2, 1, 0)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine is None:
        affine = np.eye(4)
    img = nib.nifti1.Nifti1Image(volume, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, path)


def load_nii_volume(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(path)

    dim = img.header["dim"]
    assert dim[0] == 3 or (dim[0] > 3 and all(d == 1 for d in dim[4:])), (
        "Expect a 3D volume but the input is %dD" % dim[0]
    )

    volume = img.get_fdata().astype(np.float32)
    while volume.ndim > 3:
        volume = volume.squeeze(-1)
    volume = volume.transpose(2, 1, 0)

    resolutions = img.header["pixdim"][1:4]

    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()

    return volume, resolutions, affine


def save_slices(folder: str, images: List[Slice]) -> None:
    for i, image in enumerate(images):
        image.save(os.path.join(folder, f"{i}.nii.gz"), True)


def load_slices(folder: str, device=torch.device("cpu")) -> List[Slice]:
    slices = []
    ids = []
    for f in os.listdir(folder):
        if not (f.endswith("nii") or f.endswith("nii.gz")):
            continue
        ids.append(int(f.split(".nii")[0]))
        slice, resolutions, affine = load_nii_volume(os.path.join(folder, f))
        slice_tensor = torch.tensor(slice, device=device)
        mask_tensor = slice_tensor > 0
        slice_tensor, mask_tensor, transformation = affine2transformation(
            slice_tensor, mask_tensor, resolutions, affine
        )
        slices.append(
            Slice(
                image=slice_tensor,
                mask=mask_tensor,
                transformation=transformation,
                resolution_x=resolutions[0],
                resolution_y=resolutions[1],
                resolution_z=resolutions[2],
            )
        )
    return [slice for _, slice in sorted(zip(ids, slices))]


def load_stack(
    path_vol: str, path_mask: Optional[str] = None, device=torch.device("cpu")
) -> Stack:
    slices, resolutions, affine = load_nii_volume(path_vol)
    if path_mask is None:
        mask = slices > 0
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0
        if not compare_resolution_affine(
            resolutions, affine, resolutions_m, affine_m, slices.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the input stack and stack mask do not match!"
            )

    slices_tensor = torch.tensor(slices, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    slices_tensor, mask_tensor, transformation = affine2transformation(
        slices_tensor, mask_tensor, resolutions, affine
    )

    return Stack(
        slices=slices_tensor.unsqueeze(1),
        mask=mask_tensor.unsqueeze(1),
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        thickness=resolutions[2],
        gap=resolutions[2],
    )


def load_volume(
    path_vol: str, path_mask: Optional[str] = None, device=torch.device("cpu")
) -> Volume:
    vol, resolutions, affine = load_nii_volume(path_vol)
    if path_mask is None:
        mask = vol > 0
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0
        if not compare_resolution_affine(
            resolutions, affine, resolutions_m, affine_m, vol.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the input stack and stack mask do not match!"
            )

    vol_tensor = torch.tensor(vol, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    vol_tensor, mask_tensor, transformation = affine2transformation(
        vol_tensor, mask_tensor, resolutions, affine
    )

    transformation = RigidTransform(transformation.axisangle().mean(0, keepdim=True))

    return Volume(
        image=vol_tensor,
        mask=mask_tensor,
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        resolution_z=resolutions[2],
    )
