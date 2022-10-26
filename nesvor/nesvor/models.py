from argparse import Namespace
from math import log2
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
import torch
import torch.nn.functional as F
import torch.nn as nn
import tinycudann as tcnn
from ..transform import RigidTransform, ax_transform_points, mat_transform_points
from ..utils import resolution2sigma
import logging


# key for loss/regularization
D_LOSS = "MSE"
S_LOSS = "logVar"
DS_LOSS = "MSE+logVar"
B_REG = "biasReg"
T_REG = "transReg"
I_REG = "imageReg"


class INR(nn.Module):
    def __init__(self, bounding_box: torch.Tensor, args: Namespace) -> None:
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        self.register_buffer("bounding_box", bounding_box)
        # hash grid encoding
        base_resolution = (
            (
                (self.bounding_box[1] - self.bounding_box[0]).max()
                / args.coarse_resolution
            )
            .ceil()
            .int()
            .item()
        )
        n_levels = (
            (
                torch.log2(
                    (self.bounding_box[1] - self.bounding_box[0]).max()
                    / args.finest_resolution
                    / base_resolution
                )
                / log2(args.level_scale)
                + 1
            )
            .ceil()
            .int()
            .item()
        )
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": args.n_features_per_level,
                "log2_hashmap_size": args.log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": args.level_scale,
            },
        )
        # density net
        self.density_net = tcnn.Network(
            n_input_dims=n_levels * args.n_features_per_level,
            n_output_dims=1 + args.n_features_z,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": args.width,
                "n_hidden_layers": args.depth,
            },
        )
        # logging
        logging.debug(
            "hyperparameters for hash grid encoding: "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * args.level_scale ** (n_levels - 1)),
            args.level_scale,
            n_levels,
        )
        logging.debug(
            "bounding box for reconstruction (mm): "
            + "x=[%f, %f], y=[%f, %f], z=[%f, %f]",
            self.bounding_box[0, 0],
            self.bounding_box[1, 0],
            self.bounding_box[0, 1],
            self.bounding_box[1, 1],
            self.bounding_box[0, 2],
            self.bounding_box[1, 2],
        )

    def forward(self, x: torch.Tensor, return_all: bool = True):
        x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        prefix_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        pe = self.encoding(x)
        z = self.density_net(pe)
        density = F.softplus(z[..., 0].view(prefix_shape))
        if return_all:
            return density, pe, z
        else:
            return density

    def sample_batch(
        self,
        xyz: torch.Tensor,
        transformation: Optional[RigidTransform],
        psf_sigma: Union[float, torch.Tensor],
        n_samples: int,
    ) -> torch.Tensor:
        if n_samples > 1:
            if isinstance(psf_sigma, torch.Tensor):
                psf_sigma = psf_sigma.view(-1, 1, 3)
            xyz_psf = torch.randn(
                xyz.shape[0], n_samples, 3, dtype=xyz.dtype, device=xyz.device
            )
            xyz = xyz[:, None] + xyz_psf * psf_sigma
        else:
            xyz = xyz[:, None]
        if transformation is not None:
            trans_first = transformation.trans_first
            mat = transformation.matrix(trans_first)
            xyz = mat_transform_points(mat[:, None], xyz, trans_first)
        return xyz


class NeSVoR(nn.Module):
    def __init__(
        self,
        transformation: RigidTransform,
        resolution: torch.Tensor,
        v_mean: float,
        bounding_box: torch.Tensor,
        args: Namespace,
    ) -> None:
        super().__init__()
        self.args = args
        self.n_slices = 0
        self.trans_first = True
        self.transformation = transformation
        self.psf_sigma = resolution2sigma(resolution, isotropic=False)
        self.delta = args.delta * v_mean  # 0.2 * v_mean  # 0.05*v_mean
        # self.psf_mean = 2.0/3.0
        if self.args.image_regularization == "TV":
            self.image_regularization = tv_reg
        elif self.args.image_regularization == "edge":
            self.image_regularization = edge_reg
        elif self.args.image_regularization == "L2":
            self.image_regularization = l2_reg
        self.build_network(bounding_box)
        self.to(args.device)

    @property
    def transformation(self) -> RigidTransform:
        return RigidTransform(self.axisangle.detach(), self.trans_first)

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        if self.n_slices == 0:
            self.n_slices = len(value)
        else:
            assert self.n_slices == len(value)
        axisangle = value.axisangle(self.trans_first)
        if TYPE_CHECKING:
            self.axisangle_init: torch.Tensor
        self.register_buffer("axisangle_init", axisangle.detach().clone())
        if self.args.transformation_optimization:
            self.axisangle = nn.Parameter(axisangle.detach().clone())
        else:
            self.register_buffer("axisangle", axisangle.detach().clone())

    def build_network(self, bounding_box) -> None:
        if self.args.n_features_slice:
            self.slice_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_slice
            )
        if self.args.slice_scale:
            self.logit_coef = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        if self.args.slice_variance:
            self.log_var_slice = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        # INR
        self.inr = INR(bounding_box, self.args)
        # sigma net
        if self.args.pixel_variance:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.args.n_features_slice + self.args.n_features_z,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.args.width,
                    "n_hidden_layers": self.args.depth,
                },
            )
        # bias net
        if self.args.n_levels_bias:
            self.b_net = tcnn.Network(
                n_input_dims=self.args.n_levels_bias * self.args.n_features_per_level
                + self.args.n_features_slice,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.args.width,
                    "n_hidden_layers": self.args.depth,
                },
            )

    def forward(
        self,
        xyz: torch.Tensor,
        v: torch.Tensor,
        slice_idx: torch.Tensor,
    ) -> Dict[str, Any]:
        # sample psf point
        batch_size = xyz.shape[0]
        n_samples = self.args.n_samples
        xyz_psf = torch.randn(
            batch_size, n_samples, 3, dtype=xyz.dtype, device=xyz.device
        )
        """ Sinc PSF
            r2 = (
                xyz_psf[..., 0] * xyz_psf[..., 0] + xyz_psf[..., 1] * xyz_psf[..., 1]
            )  # x^2+y^2
            psf = torch.sinc(torch.sqrt(r2) * (1.2068 / 2.3548)) ** 2 / torch.exp(
                -0.5 * r2
            )  # * 1.5 #* 0.4
            self.psf_mean = self.psf_mean * 0.95 + psf.mean().detach() * 0.05
            psf = torch.clamp(psf / self.psf_mean, 0.0, 5.0 / self.psf_mean * 2)
        """
        psf = 1
        psf_sigma = self.psf_sigma[slice_idx][:, None]
        # transform points
        t = self.axisangle[slice_idx][:, None]
        xyz = ax_transform_points(
            t, xyz[:, None] + xyz_psf * psf_sigma, self.trans_first
        )
        # inputs
        if self.args.n_features_slice:
            se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
        else:
            se = None
        # forward
        results = self.net_forward(xyz, se)
        # output
        density = results["density"]
        if "log_bias" in results:
            log_bias = results["log_bias"]
            bias = log_bias.exp()
            bias_detach = bias.detach()
        else:
            log_bias = 0
            bias = 1
            bias_detach = 1
        if "log_var" in results:
            log_var = results["log_var"]
            var = log_var.exp()
        else:
            log_var = 0
            var = 1
        # imaging
        if self.args.slice_scale:
            c: Any = F.softmax(self.logit_coef, 0)[slice_idx] * self.n_slices
        else:
            c = 1
        v_out = (bias * density).mean(-1)
        v_out = c * v_out
        if self.args.pixel_variance:
            # var = (bias_detach ** 2) * var
            # var = (c.detach() ** 2  / n_samples) * var.mean(-1)
            var = (bias_detach * psf * var).mean(-1)
            var = c.detach() * var
            var = var**2
        if self.args.slice_variance:
            var = var + self.log_var_slice.exp()[slice_idx]
        # losses
        losses = {D_LOSS: ((v_out - v) ** 2 / (2 * var)).mean()}
        if self.args.pixel_variance or self.args.slice_variance:
            losses[S_LOSS] = 0.5 * var.log().mean()
            losses[DS_LOSS] = losses[D_LOSS] + losses[S_LOSS]
        if self.args.transformation_optimization:
            losses[T_REG] = self.trans_loss(trans_first=self.trans_first)
        if self.args.n_levels_bias:
            losses[B_REG] = log_bias.mean() ** 2
        # image regularization
        losses[I_REG] = self.image_regularization(density, xyz, self.delta)

        return losses

    def net_forward(
        self,
        x: torch.Tensor,
        se: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        density, pe, z = self.inr(x)
        prefix_shape = density.shape
        results = {"density": density}

        zs = []
        if se is not None:
            zs.append(se.reshape(-1, se.shape[-1]))

        if self.args.n_levels_bias:
            pe_bias = pe[
                ..., : self.args.n_levels_bias * self.args.n_features_per_level
            ]
            results["log_bias"] = self.b_net(torch.cat(zs + [pe_bias], -1)).view(
                prefix_shape
            )

        if self.args.pixel_variance:
            zs.append(z[..., 1:])
            results["log_var"] = self.sigma_net(torch.cat(zs, -1)).view(prefix_shape)

        return results

    def trans_loss(self, trans_first: bool = True) -> torch.Tensor:
        x = RigidTransform(self.axisangle, trans_first=trans_first)
        y = RigidTransform(self.axisangle_init, trans_first=trans_first)
        err = y.inv().compose(x).axisangle(trans_first=trans_first)
        loss_R = torch.mean(err[:, :3] ** 2)
        loss_T = torch.mean(err[:, 3:] ** 2)
        return loss_R + 1e-3 * loss_T


def tv_reg(density: torch.Tensor, xyz: torch.Tensor, delta: float):
    d_density = density - torch.flip(density, (1,))
    dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
    dd_dx = d_density / dx2.sqrt()
    return torch.abs(dd_dx).mean()


def edge_reg(density: torch.Tensor, xyz: torch.Tensor, delta: float):
    d_density = density - torch.flip(density, (1,))
    dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
    dd2_dx2 = d_density**2 / dx2 / (delta * delta)
    return delta * ((1 + dd2_dx2).sqrt().mean() - 1)


def l2_reg(density: torch.Tensor, xyz: torch.Tensor, delta: float):
    d_density = density - torch.flip(density, (1,))
    dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
    dd2_dx2 = d_density**2 / dx2
    return dd2_dx2.mean()
