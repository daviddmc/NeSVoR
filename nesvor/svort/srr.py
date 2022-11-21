import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transform import axisangle2mat
from ..slice_acquisition import slice_acquisition, slice_acquisition_adjoint


def dot(x, y):
    return torch.dot(x.flatten(), y.flatten())


def CG(A, b, x0, n_iter, tol=0.0):
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        if dot_r_r_new <= tol:
            return x
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new


def PSFreconstruction(transforms, slices, slices_mask, vol_mask, params):
    return slice_acquisition_adjoint(
        transforms,
        params["psf"],
        slices,
        slices_mask,
        vol_mask,
        params["volume_shape"],
        params["res_s"] / params["res_r"],
        params["interp_psf"],
        True,
    )


class SRR(nn.Module):
    def __init__(
        self, n_iter=10, use_CG=False, alpha=0.5, beta=0.02, delta=0.1, tol=0.0
    ):
        super().__init__()
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta * delta * delta
        self.delta = delta
        self.use_CG = use_CG
        self.tol = tol

    def forward(
        self,
        theta,
        slices,
        volume,
        params,
        p=None,
        mu=0,
        z=None,
        vol_mask=None,
        slices_mask=None,
    ):
        if len(theta.shape) == 2:
            transforms = axisangle2mat(theta)
        else:
            transforms = theta

        A = lambda x: self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x: self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x: self.AtA(transforms, x, vol_mask, slices_mask, p, params, mu, z)

        x = volume
        y = slices

        if self.use_CG:
            b = At(y * p if p is not None else y)
            if mu and z is not None:
                b = b + mu * z
            x = CG(AtA, b, volume, self.n_iter, self.tol)
        else:
            for _ in range(self.n_iter):
                err = A(x) - y
                if p is not None:
                    err = p * err
                g = At(err)
                if self.beta:
                    dR = self.dR(x, self.delta)
                    g.add_(dR, alpha=self.beta)
                x.add_(g, alpha=-self.alpha)
        return F.relu(x, True)

    def A(self, transforms, x, vol_mask, slices_mask, params):
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            params["interp_psf"],
        )

    def At(self, transforms, x, slices_mask, vol_mask, params):
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            params["interp_psf"],
            False,
        )

    def AtA(self, transforms, x, vol_mask, slices_mask, p, params, mu, z):
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if mu and z is not None:
            vol = vol + mu * x
        return vol

    def dR(self, v, delta):
        g = torch.zeros_like(v)
        D, H, W = v.shape[-3:]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    v0 = v[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1]
                    v1 = v[
                        :,
                        :,
                        1 + dz : D - 1 + dz,
                        1 + dy : H - 1 + dy,
                        1 + dx : W - 1 + dx,
                    ]
                    dv = v0 - v1
                    dv_ = dv * (1 / (dx * dx + dy * dy + dz * dz) / (delta * delta))
                    g[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1] += dv_ / torch.sqrt(
                        1 + dv * dv_
                    )
        return g
