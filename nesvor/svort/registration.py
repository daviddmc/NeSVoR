import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transform import RigidTransform
from ..utils import ncc_loss, gaussian_blur, meshgrid
import numpy as np
import types


class Registration(nn.Module):
    def __init__(
        self, num_levels, num_steps, step_size, max_iter, optimizer, loss, auto_grad
    ):
        super().__init__()
        self.num_levels = num_levels
        self.current_level = self.num_levels - 1
        self.num_steps = [num_steps] * self.num_levels
        self.step_sizes = [step_size * 2**level for level in range(num_levels)]
        self.max_iter = max_iter
        self.auto_grad = auto_grad
        self._degree2rad = None

        # init loss
        if isinstance(loss, dict):
            loss_name = loss.pop("name")
            if loss_name == "mse":
                self.loss = types.MethodType(
                    lambda s, x, y: F.mse_loss(x, y, reduction="none", **loss), self
                )
            elif loss_name == "ncc":
                self.loss = types.MethodType(
                    lambda s, x, y: ncc_loss(
                        x, y, reduction="none", level=s.current_level, **loss
                    ),
                    self,
                )
            else:
                raise Exception("unknown loss")
        elif callable(loss):
            self.loss = types.MethodType(lambda s, x, y: loss(s, x, y), self)
        else:
            raise Exception("unknown loss")

        # init optimizer
        if optimizer["name"] == "gd":
            if "momentum" not in optimizer:
                optimizer["momentum"] = 0
        self.optimizer = optimizer

    def degree2rad(self, theta):
        if self._degree2rad is None:
            self._degree2rad = torch.tensor(
                [np.pi / 180, np.pi / 180, np.pi / 180, 1, 1, 1],
                device=theta.device,
                dtype=theta.dtype,
            ).view(1, 6)
        return theta * self._degree2rad

    def rad2degree(self, theta):
        if self._degree2rad is None:
            self._degree2rad = torch.tensor(
                [np.pi / 180, np.pi / 180, np.pi / 180, 1, 1, 1],
                device=theta.device,
                dtype=theta.dtype,
            ).view(1, 6)
        return theta / self._degree2rad

    def clean_optimizer_state(self):
        if self.optimizer["name"] == "gd":
            if "buf" in self.optimizer:
                self.optimizer.pop("buf")

    def prepare(self, *args, **kwargs):
        return

    def forward(self, theta, source, target, params):
        self.prepare(theta, source, target, params)
        theta0 = theta.clone()
        theta = self.rad2degree(theta.detach()).requires_grad_(self.auto_grad)
        with torch.set_grad_enabled(self.auto_grad):
            theta, loss = self.multilevel(theta, source, target)
        with torch.no_grad():
            dtheta = self.degree2rad(theta) - theta0
        return theta0 + dtheta, loss

    def update_level(self, theta, source, target):
        raise NotImplementedError("")

    def multilevel(self, theta, source, target):
        for level in range(self.num_levels - 1, -1, -1):
            self.current_level = level
            source_new, target_new = self.update_level(theta, source, target)
            theta, loss = self.singlelevel(
                theta,
                source_new,
                target_new,
                self.num_steps[level],
                self.step_sizes[level],
            )
            self.clean_optimizer_state()

        return theta, loss

    def singlelevel(self, theta, source, target, num_steps, step_size):
        for _ in range(num_steps):
            theta, loss = self.step(theta, source, target, step_size)
            step_size /= 2
        return theta, loss

    def step(self, theta, source, target, step_size):
        self.activate_idx = torch.ones(
            theta.shape[0], device=theta.device, dtype=torch.bool
        )
        loss_all = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
        for _ in range(self.max_iter):
            theta_a, source_a, target_a = self.activate_set(theta, source, target)
            loss, grad = self.grad(theta_a, source_a, target_a, step_size)
            loss_all[self.activate_idx] = loss

            with torch.no_grad():
                step = self.optimizer_step(grad) * -step_size
                theta_a.add_(step)
                loss_new = self.evaluate(theta_a, source_a, target_a)
                idx_new = loss_new < loss
                self.activate_idx[self.activate_idx.clone()] = idx_new
                if not torch.any(self.activate_idx):
                    break
                theta[self.activate_idx] += step[idx_new]

        return theta, loss_all.detach()

    def activate_set(self, theta, source, target):
        theta = theta[self.activate_idx]
        if source.shape[0] > 1:
            source = source[self.activate_idx]
        if target.shape[0] > 1:
            target = target[self.activate_idx]
        return theta, source, target

    def grad(self, theta, source, target, step_size):
        loss = self.evaluate(theta, source, target)
        if self.auto_grad:
            grad = torch.autograd.grad([loss.sum()], [theta])[0]
        else:
            backup = torch.empty_like(theta[:, 0])
            grad = torch.zeros_like(theta)
            for j in range(theta.shape[1]):
                backup.copy_(theta[:, j])
                theta[:, j].copy_(backup + step_size)
                loss1 = self.evaluate(theta, source, target)
                theta[:, j].copy_(backup - step_size)
                loss2 = self.evaluate(theta, source, target)
                theta[:, j].copy_(backup)
                grad[:, j] = loss1 - loss2
        return loss, grad

    def warp(self, theta, source, target):
        raise NotImplementedError("warp")

    def evaluate(self, theta, source, target):
        warpped, target = self.warp(theta, source, target)
        loss = self.loss(warpped, target)
        loss = loss.view(loss.shape[0], -1).mean(1)
        return loss

    def optimizer_step(self, grad):
        if self.optimizer["name"] == "gd":
            step = self.gd_step(grad)
        else:
            raise Exception("unknown optimizer")
        step = step / (torch.linalg.norm(step, dim=-1, keepdim=True) + 1e-6)
        return step

    def gd_step(self, grad):
        if self.optimizer["momentum"]:
            if "buf" not in self.optimizer:
                self.optimizer["buf"] = grad.clone()
            else:
                self.optimizer["buf"][self.activate_idx] = (
                    self.optimizer["buf"][self.activate_idx]
                    * self.optimizer["momentum"]
                    + grad
                )
            return self.optimizer["buf"][self.activate_idx]
        else:
            return grad


class VVR(Registration):
    def __init__(
        self, num_levels, num_steps, step_size, max_iter, optimizer, loss, auto_grad
    ):
        super().__init__(
            num_levels, num_steps, step_size, max_iter, optimizer, loss, auto_grad
        )
        self.theta_t = None
        self._grid = None
        self._grid_scale = None
        self._target_flat = None
        self.trans_first = True

    def update_level(self, theta, source, target):
        sigma = [0.5 * (2**self.current_level) / res for res in self.relative_res]
        source = gaussian_blur(source, sigma, truncated=4.0)
        target = gaussian_blur(target, sigma, truncated=4.0)

        source = resample(
            source, self.relative_res[::-1], [2**self.current_level] * 3
        )  # self.resample(source)
        target = resample(
            target, self.relative_res[::-1], [2**self.current_level] * 3
        )  # self.resample(target)

        res_new = self.res * (2**self.current_level)
        mask = (target > 0).view(-1)

        grid = meshgrid(
            (target.shape[-1], target.shape[-2], target.shape[-3]),
            (res_new, res_new, res_new),
            device=target.device,
        )
        grid = grid.reshape(-1, 3)[mask, :]
        self._grid = grid

        self._target_flat = target.view(-1)[mask]

        scale = torch.tensor(
            [
                2.0 / (source.shape[-1] - 1),
                2.0 / (source.shape[-2] - 1),
                2.0 / (source.shape[-3] - 1),
            ],
            device=source.device,
            dtype=source.dtype,
        )
        self._grid_scale = scale / res_new

        return source, target

    def warp(self, theta, source, target):
        transforms = (
            RigidTransform(self.degree2rad(theta), trans_first=self.trans_first)
            .inv()
            .compose(self.theta_t)
            .matrix()
        )

        grid = torch.matmul(
            transforms[:, :, :-1], self._grid.reshape(-1, 3, 1) + transforms[:, :, -1:]
        )
        grid = grid.reshape(1, -1, 1, 1, 3)
        warpped = F.grid_sample(source, grid * self._grid_scale, align_corners=True)

        return warpped.view(1, 1, -1), self._target_flat.view(1, 1, -1)

    def prepare(self, theta, source, target, params):
        res = [params["s_thick"], params["res_s"], params["res_s"]]
        self.res = min(res)
        self.relative_res = [r / self.res for r in res]

    def __call__(self, theta, source, target, params, transform_t, trans_first):
        self.theta_t = transform_t
        self.trans_first = trans_first
        return super().__call__(theta, source, target, params)


def resample(x, res_xyz_old, res_xyz_new):
    ndim = x.ndim - 2
    assert len(res_xyz_new) == len(res_xyz_old) == ndim
    grids = []
    for i in range(ndim):
        fac = res_xyz_old[i] / res_xyz_new[i]
        size_new = int(x.shape[-i - 1] * fac)
        grid_max = (size_new - 1) / fac / (x.shape[-i - 1] - 1)
        grids.append(
            torch.linspace(
                -grid_max, grid_max, size_new, dtype=x.dtype, device=x.device
            )
        )
    grid = torch.stack(torch.meshgrid(*grids[::-1], indexing="ij")[::-1], -1)
    y = F.grid_sample(
        x, grid[None].expand((x.shape[0],) + (-1,) * (ndim + 1)), align_corners=True
    )
    return y
