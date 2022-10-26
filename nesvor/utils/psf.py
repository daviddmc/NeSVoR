from typing import List, Tuple, Optional
import torch
from math import log, sqrt

GAUSSIAN_FWHM = 1 / (2 * sqrt(2 * log(2)))
SINC_FWHM = 1.206709128803223 * GAUSSIAN_FWHM


def resolution2sigma(rx, ry=None, rz=None, /, isotropic=False):
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM
        fz = GAUSSIAN_FWHM
    assert not ((ry is None) ^ (rz is None))
    if ry is None:
        if isinstance(rx, float) or isinstance(rx, int):
            if isotropic:
                return fx * rx
            else:
                return fx * rx, fy * rx, fz * rx
        elif isinstance(rx, torch.Tensor):
            if isotropic:
                return fx * rx
            else:
                assert rx.shape[-1] == 3
                return rx * torch.tensor([fx, fy, fz], dtype=rx.dtype, device=rx.device)
        elif isinstance(rx, List) or isinstance(rx, Tuple):
            assert len(rx) == 3
            return resolution2sigma(rx[0], rx[1], rx[2], isotropic=isotropic)
        else:
            raise Exception(str(type(rx)))
    else:
        return fx * rx, fy * ry, fz * rz


def get_PSF(
    r_max: Optional[int] = None,
    res_ratio: Tuple[float, float, float] = (1, 1, 3),
    threshold: float = 1e-3,
    device=torch.device("cpu"),
) -> torch.Tensor:
    sigma_x, sigma_y, sigma_z = resolution2sigma(res_ratio, isotropic=False)
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    psf = torch.exp(
        -0.5
        * (
            grid_x**2 / sigma_x**2
            + grid_y**2 / sigma_y**2
            + grid_z**2 / sigma_z**2
        )
    )
    psf[psf.abs() < threshold] = 0
    rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item())
    ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
    rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()
    psf = psf / psf.sum()
    return psf
