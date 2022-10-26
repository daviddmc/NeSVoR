from typing import Optional
import torch
import torch.nn.functional as F


def ncc_loss(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    win: Optional[int] = 9,
    level: int = 0,
    eps: float = 1e-6,
    reduction: str = "none",
) -> torch.Tensor:
    spatial_dims = len(I.shape) - 2

    if mask is not None:
        I = I * mask
        J = J * mask

    c = I.shape[1]

    if win is None:
        I = torch.flatten(I, 1)
        J = torch.flatten(J, 1)
        if mask is not None:
            mask = torch.flatten(mask, 1)
            N = mask.sum(-1) + eps
            I_mean = I.sum(-1) / N
            J_mean = J.sum(-1) / N
            I2_mean = (I * I).sum(-1) / N
            J2_mean = (J * J).sum(-1) / N
            IJ_mean = (I * J).sum(-1) / N
        else:
            I_mean = I.mean(-1)
            J_mean = J.mean(-1)
            I2_mean = (I * I).mean(-1)
            J2_mean = (J * J).mean(-1)
            IJ_mean = (I * J).mean(-1)
    else:
        I = I.view(-1, 1, *I.shape[2:])
        J = J.view(-1, 1, *J.shape[2:])

        win = 2 * int(win / 2**level / 2) + 1

        mean_filt = torch.ones([1, 1] + [win] * spatial_dims, device=I.device) / (
            win**spatial_dims
        )
        conv_fn = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]

        I_mean = conv_fn(I, mean_filt, stride=1, padding=win // 2)
        J_mean = conv_fn(J, mean_filt, stride=1, padding=win // 2)
        I2_mean = conv_fn(I * I, mean_filt, stride=1, padding=win // 2)
        J2_mean = conv_fn(J * J, mean_filt, stride=1, padding=win // 2)
        IJ_mean = conv_fn(I * J, mean_filt, stride=1, padding=win // 2)

    cross = IJ_mean - I_mean * J_mean
    I_var = I2_mean - I_mean * I_mean
    J_var = J2_mean - J_mean * J_mean

    cc = cross * cross / (I_var * J_var + eps)

    if reduction == "mean":
        return -cc.mean()
    elif reduction == "sum":
        return -cc.sum()
    else:
        if win is None:
            return -cc.view(-1, c)
        else:
            return -cc.view(-1, c, *I.shape[2:])
