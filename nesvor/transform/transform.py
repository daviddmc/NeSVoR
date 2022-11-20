from __future__ import annotations
from typing import Iterable
import torch
import numpy as np
from .transform_convert import axisangle2mat, mat2axisangle


class RigidTransform(object):
    def __init__(
        self, data: torch.Tensor, trans_first: bool = True, device=None
    ) -> None:
        self.trans_first = trans_first
        self._axisangle = None
        self._matrix = None
        if device is not None:
            data = data.to(device)
        if data.shape[1] == 6 and data.ndim == 2:  # parameter
            self._axisangle = data
        elif data.shape[1] == 3 and data.ndim == 3:  # matrix
            self._matrix = data
        else:
            raise Exception("Unknown format for rigid transform!")

    def matrix(self, trans_first: bool = True) -> torch.Tensor:
        if self._matrix is not None:
            mat = self._matrix
        else:
            mat = axisangle2mat(self._axisangle)
        if self.trans_first == True and trans_first == False:
            mat = mat_first2last(mat)
        elif self.trans_first == False and trans_first == True:
            mat = mat_last2first(mat)
        return mat

    def axisangle(self, trans_first: bool = True) -> torch.Tensor:
        if self._axisangle is not None:
            ax = self._axisangle
        else:
            ax = mat2axisangle(self._matrix)
        if self.trans_first == True and trans_first == False:
            ax = ax_first2last(ax)
        elif self.trans_first == False and trans_first == True:
            ax = ax_last2first(ax)
        return ax

    def inv(self) -> RigidTransform:
        mat = self.matrix(trans_first=True)
        R = mat[:, :, :3]
        t = mat[:, :, 3:]
        mat = torch.cat((R.transpose(-2, -1), -torch.matmul(R, t)), -1)
        return RigidTransform(mat, trans_first=True)

    def compose(self, other: RigidTransform) -> RigidTransform:
        mat1 = self.matrix(trans_first=True)
        mat2 = other.matrix(trans_first=True)
        R1 = mat1[:, :, :3]
        t1 = mat1[:, :, 3:]
        R2 = mat2[:, :, :3]
        t2 = mat2[:, :, 3:]
        R = torch.matmul(R1, R2)
        t = t2 + torch.matmul(R2.transpose(-2, -1), t1)
        mat = torch.cat((R, t), -1)
        return RigidTransform(mat, trans_first=True)

    def __getitem__(self, idx) -> RigidTransform:
        if self._axisangle is not None:
            data = self._axisangle[idx]
            if len(data.shape) < 2:
                data = data.unsqueeze(0)
        elif self._matrix is not None:
            data = self._matrix[idx]
            if len(data.shape) < 3:
                data = data.unsqueeze(0)
        else:
            raise Exception("Both data are None!")
        return RigidTransform(data, self.trans_first)

    def detach(self) -> RigidTransform:
        if self._axisangle is not None:
            data = self._axisangle.detach()
        elif self._matrix is not None:
            data = self._matrix.detach()
        else:
            raise Exception("Both data are None!")
        return RigidTransform(data, self.trans_first)

    def clone(self) -> RigidTransform:
        if self._axisangle is not None:
            data = self._axisangle.clone()
        elif self._matrix is not None:
            data = self._matrix.clone()
        else:
            raise Exception("Both data are None!")
        return RigidTransform(data, self.trans_first)

    @property
    def device(self):
        if self._axisangle is not None:
            return self._axisangle.device
        elif self._matrix is not None:
            return self._matrix.device
        else:
            raise Exception("Both data are None!")

    @staticmethod
    def cat(transforms: Iterable[RigidTransform]) -> RigidTransform:
        matrixs = [t.matrix(trans_first=True) for t in transforms]
        return RigidTransform(torch.cat(matrixs, 0), trans_first=True)

    def __len__(self) -> int:
        if self._axisangle is not None:
            return self._axisangle.shape[0]
        elif self._matrix is not None:
            return self._matrix.shape[0]
        else:
            raise Exception("Both data are None!")


def mat_first2last(mat: torch.Tensor) -> torch.Tensor:
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(R, t)
    mat = torch.cat([R, t], -1)
    return mat


def mat_last2first(mat: torch.Tensor) -> torch.Tensor:
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(R.transpose(-2, -1), t)
    mat = torch.cat([R, t], -1)
    return mat


def ax_first2last(axisangle: torch.Tensor) -> torch.Tensor:
    mat = axisangle2mat(axisangle)
    mat = mat_first2last(mat)
    return mat2axisangle(mat)


def ax_last2first(axisangle: torch.Tensor) -> torch.Tensor:
    mat = axisangle2mat(axisangle)
    mat = mat_last2first(mat)
    return mat2axisangle(mat)


def mat_update_resolution(mat: torch.Tensor, res_from, res_to) -> torch.Tensor:
    assert mat.dim() == 3
    fac = torch.ones_like(mat[:1, :1])
    fac[..., 3] = res_from / res_to
    return mat * fac


def ax_update_resolution(ax: torch.Tensor, res_from, res_to) -> torch.Tensor:
    assert ax.dim() == 2
    fac = torch.ones_like(ax[:1])
    fac[:, 3:] = res_from / res_to
    return ax * fac


def mat2euler(mat: torch.Tensor) -> torch.Tensor:
    TOL = 0.000001
    TX = mat[:, 0, 3]
    TY = mat[:, 1, 3]
    TZ = mat[:, 2, 3]

    tmp = torch.asin(-mat[:, 0, 2])
    mask = torch.cos(tmp).abs() <= TOL
    RX = torch.atan2(mat[:, 1, 2], mat[:, 2, 2])
    RY = tmp
    RZ = torch.atan2(mat[:, 0, 1], mat[:, 0, 0])
    RX[mask] = torch.atan2(-mat[:, 0, 2] * mat[:, 1, 0], -mat[:, 0, 2] * mat[:, 2, 0])[
        mask
    ]
    RZ[mask] = 0

    RX *= 180 / np.pi
    RY *= 180 / np.pi
    RZ *= 180 / np.pi

    return torch.stack((TX, TY, TZ, RX, RY, RZ), -1)


def euler2mat(p: torch.Tensor) -> torch.Tensor:
    tx = p[:, 0]
    ty = p[:, 1]
    tz = p[:, 2]

    rx = p[:, 3]
    ry = p[:, 4]
    rz = p[:, 5]

    M_PI = np.pi
    cosrx = torch.cos(rx * (M_PI / 180.0))
    cosry = torch.cos(ry * (M_PI / 180.0))
    cosrz = torch.cos(rz * (M_PI / 180.0))
    sinrx = torch.sin(rx * (M_PI / 180.0))
    sinry = torch.sin(ry * (M_PI / 180.0))
    sinrz = torch.sin(rz * (M_PI / 180.0))

    mat = torch.eye(4, device=p.device)
    mat = mat.reshape((1, 4, 4)).repeat(p.shape[0], 1, 1)

    mat[:, 0, 0] = cosry * cosrz
    mat[:, 0, 1] = cosry * sinrz
    mat[:, 0, 2] = -sinry
    mat[:, 0, 3] = tx

    mat[:, 1, 0] = sinrx * sinry * cosrz - cosrx * sinrz
    mat[:, 1, 1] = sinrx * sinry * sinrz + cosrx * cosrz
    mat[:, 1, 2] = sinrx * cosry
    mat[:, 1, 3] = ty

    mat[:, 2, 0] = cosrx * sinry * cosrz + sinrx * sinrz
    mat[:, 2, 1] = cosrx * sinry * sinrz - sinrx * cosrz
    mat[:, 2, 2] = cosrx * cosry
    mat[:, 2, 3] = tz
    mat[:, 3, 3] = 1.0

    return mat[:, :3, :]


def point2mat(p: torch.Tensor) -> torch.Tensor:
    p = p.view(-1, 3, 3)
    p1 = p[:, 0]
    p2 = p[:, 1]
    p3 = p[:, 2]
    v1 = p3 - p1
    v2 = p2 - p1

    nz = torch.cross(v1, v2, -1)
    ny = torch.cross(nz, v1, -1)
    nx = v1

    R = torch.stack((nx, ny, nz), -1)
    R = R / torch.linalg.norm(R, ord=2, dim=-2, keepdim=True)

    T = torch.matmul(R.transpose(-2, -1), p2.unsqueeze(-1))

    return torch.cat((R, T), -1)


def mat2point(mat: torch.Tensor, sx, sy, rs) -> torch.Tensor:
    p1 = torch.tensor([-(sx - 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(
        dtype=mat.dtype, device=mat.device
    )
    p2 = torch.tensor([0, 0, 0]).to(dtype=mat.dtype, device=mat.device)
    p3 = torch.tensor([(sx - 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(
        dtype=mat.dtype, device=mat.device
    )
    p = torch.stack((p1, p2, p3), 0)
    p = p.unsqueeze(0).unsqueeze(-1)  # 1x3x3x1
    R = mat[:, :, :-1].unsqueeze(1)  # nx1x3x3
    T = mat[:, :, -1:].unsqueeze(1)  # nx1x3x1
    p = torch.matmul(R, p + T)
    return p.view(-1, 9)


def mat_transform_points(
    mat: torch.Tensor, x: torch.Tensor, trans_first: bool
) -> torch.Tensor:
    # mat (*, 3, 4)
    # x (*, 3)
    R = mat[..., :-1]  # (*, 3, 3)
    T = mat[..., -1:]  # (*, 3, 1)
    x = x[..., None]  # (*, 3, 1)
    if trans_first:
        x = torch.matmul(R, x + T)  # (*, 3)
    else:
        x = torch.matmul(R, x) + T
    return x[..., 0]


def ax_transform_points(
    ax: torch.Tensor, x: torch.Tensor, trans_first: bool
) -> torch.Tensor:
    # ax (*, 6)
    # x (*, 3)
    mat = axisangle2mat(ax.view(-1, 6)).view(ax.shape[:-1] + (3, 4))
    return mat_transform_points(mat, x, trans_first)


def transform_points(transform: RigidTransform, x: torch.Tensor) -> torch.Tensor:
    # transform (N) and x (N, 3)
    # or transform (1) and x (*, 3)
    assert x.ndim == 2 and x.shape[-1] == 3
    trans_first = transform.trans_first
    mat = transform.matrix(trans_first)
    return mat_transform_points(mat, x, trans_first)
