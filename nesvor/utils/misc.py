from typing import Dict, List, Any, Optional, Union, Collection, Iterable
import torch
import torch.nn.functional as F
import collections
from argparse import Namespace
import os


def makedirs(path: Union[str, Iterable[str]]) -> None:
    if isinstance(path, str):
        path = [path]
    for p in path:
        if p:
            try:
                os.makedirs(p, exist_ok=False)
            except FileExistsError:
                pass
            except Exception as e:
                raise e


def merge_args(args_old: Namespace, args_new: Namespace) -> Namespace:
    dict_old = vars(args_old)
    dict_new = vars(args_new)
    dict_old.update(dict_new)
    return Namespace(**dict_old)


def meshgrid(
    shape_xyz: Collection,
    resolution_xyz: Collection,
    min_xyz: Optional[Collection] = None,
    device=None,
    stack_output: bool = True,
):

    assert len(shape_xyz) == len(resolution_xyz)
    if min_xyz is None:
        min_xyz = tuple(-(s - 1) * r / 2 for s, r in zip(shape_xyz, resolution_xyz))
    else:
        assert len(shape_xyz) == len(min_xyz)

    if device is None:
        if isinstance(shape_xyz, torch.Tensor):
            device = shape_xyz.device
        elif isinstance(resolution_xyz, torch.Tensor):
            device = resolution_xyz.device
        else:
            device = torch.device("cpu")
    dtype = torch.float32

    arr_xyz = [
        torch.arange(s, dtype=dtype, device=device) * r + m
        for s, r, m in zip(shape_xyz, resolution_xyz, min_xyz)
    ]
    grid_xyz = torch.meshgrid(arr_xyz[::-1], indexing="ij")[::-1]
    if stack_output:
        return torch.stack(grid_xyz, -1)
    else:
        return grid_xyz


def gaussian_blur(
    x: torch.Tensor, sigma: Union[float, collections.abc.Iterable], truncated: float
) -> torch.Tensor:
    spatial_dims = len(x.shape) - 2
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = [sigma] * spatial_dims
    kernels = [gaussian_1d_kernel(s, truncated, x.device) for s in sigma]
    c = x.shape[1]
    conv_fn = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
    for d in range(spatial_dims):
        s = [1] * len(x.shape)
        s[d + 2] = -1
        k = kernels[d].reshape(s).repeat(*([c, 1] + [1] * spatial_dims))
        padding = [0] * spatial_dims
        padding[d] = (k.shape[d + 2] - 1) // 2
        x = conv_fn(x, k, padding=padding, groups=c)
    return x


# from MONAI
def gaussian_1d_kernel(sigma: float, truncated: float, device) -> torch.Tensor:
    tail = int(max(sigma * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
    t = 0.70710678 / sigma
    kernel = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    return kernel.clamp(min=0)


class MovingAverage:
    def __init__(self, alpha: float) -> None:
        assert 0 <= alpha < 1
        self.alpha = alpha
        self._value: Dict[str, Any] = dict()

    def to_dict(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "value": self._value}

    def from_dict(self, d: Dict) -> None:
        self.alpha = d["alpha"]
        self._value = d["value"]

    def __getitem__(self, key: str) -> Any:
        if key not in self._value:
            return 0
        num, v = self._value[key]
        if self.alpha:
            return v / (1 - self.alpha**num)
        else:
            return v / num

    def __call__(self, key: str, value) -> None:
        if key not in self._value:
            self._value[key] = (0, 0)
        num, v = self._value[key]
        num += 1
        if self.alpha:
            v = v * self.alpha + value * (1 - self.alpha)
        else:
            v += value
        self._value[key] = (num, v)

    def __str__(self) -> str:
        s = ""
        for key in self._value:
            s += "%s = %.3e  " % (key, self[key])
        if len(self._value) > 0:
            return ("iter = %d  " % self._value[key][0]) + s
        else:
            return s

    @property
    def header(self) -> str:
        return "iter," + ",".join(self._value.keys())

    @property
    def value(self) -> List:
        values = []
        for key in self._value:
            values.append(self[key])
        if len(self._value) > 0:
            return [self._value[key][0]] + values
        else:
            return values
