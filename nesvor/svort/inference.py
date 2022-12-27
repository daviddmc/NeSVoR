import logging
import time
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from .registration import VVR, resample
from .srr import PSFreconstruction, SRR
from . import SVoRT, SVoRTv2
from ..transform import RigidTransform, mat_update_resolution
from ..utils import get_PSF, ncc_loss
from ..slice_acquisition import slice_acquisition
from ..image import Stack, Slice
from .. import __checkpoint_dir, __pretrained_svort


def compute_score(ncc, ncc_weight):
    ncc_weight = ncc_weight.view(ncc.shape)
    return -((ncc * ncc_weight).sum() / ncc_weight.sum()).item()


def get_transform_diff_mean(transform_out, transform_in, mean_r=3):
    transform_diff = transform_out.compose(transform_in.inv())
    transform_diff_ax = transform_diff.axisangle()
    mid = transform_diff_ax.shape[0] // 2
    meanT = transform_diff_ax[mid - mean_r : mid + mean_r, 3:].mean(0, keepdim=True)
    meanR = average_rotation(transform_diff_ax[mid - 3 : mid + 3, :3])
    transform_diff_mean = torch.cat((meanR, meanT), -1)
    return RigidTransform(transform_diff_mean), transform_diff


def average_rotation(R):
    import scipy
    from scipy.spatial.transform import Rotation

    dtype = R.dtype
    device = R.device
    Rmat = Rotation.from_rotvec(R.cpu().numpy()).as_matrix()
    R = Rotation.from_rotvec(R.cpu().numpy()).as_quat()
    for i in range(R.shape[0]):
        if np.linalg.norm(R[i] + R[0]) < np.linalg.norm(R[i] - R[0]):
            R[i] *= -1
    barR = np.mean(R, 0)
    barR = barR / np.linalg.norm(barR)

    S_new = S = Rotation.from_quat(barR).as_matrix()
    R = Rmat
    i = 0
    while np.all(np.isreal(S_new)) and i < 10:
        S = S_new
        i += 1
        sum_vmatrix_normed = np.zeros((3, 3))
        sum_inv_norm_vmatrix = 0
        for j in range(R.shape[0]):
            vmatrix = scipy.linalg.logm(np.matmul(R[j], np.linalg.inv(S)))
            vmatrix_normed = vmatrix / np.linalg.norm(vmatrix, ord=2, axis=(0, 1))
            sum_vmatrix_normed += vmatrix_normed
            sum_inv_norm_vmatrix += 1 / np.linalg.norm(vmatrix, ord=2, axis=(0, 1))

        delta = sum_vmatrix_normed / sum_inv_norm_vmatrix
        S_new = np.matmul(scipy.linalg.expm(delta), S)

    S = Rotation.from_matrix(S).as_rotvec()
    return torch.tensor(S[None], dtype=dtype, device=device)


def run_model(transforms, stacks, model, res_s, s_thick, res_r):
    # run models
    device = stacks[0].device
    slice_shape = stacks[0].shape[-2:]
    positions = [
        torch.arange(slices.shape[0], dtype=slices.dtype, device=device)
        - slices.shape[0] // 2
        for slices in stacks
    ]

    transforms_out = []
    with torch.no_grad():
        n_run = max(1, len(stacks) - 2)
        for j in range(n_run):
            idxes = [0, 1, j + 2] if j > 0 else list(range(min(3, len(stacks))))
            volume_shape = (256, 256, 256)
            data = {
                "psf_rec": get_PSF(
                    res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
                    device=device,
                ),
                "slice_shape": slice_shape,  # (128, 128)
                "resolution_slice": res_s,
                "resolution_recon": res_r,
                "slice_thickness": s_thick,
                "volume_shape": volume_shape,  # (125, 169, 145),
                "transforms": RigidTransform.cat(
                    [transforms[idx] for idx in idxes]
                ).matrix(),
                "stacks": torch.cat([stacks[idx] for idx in idxes], dim=0),
                "positions": torch.cat(
                    [
                        torch.stack(
                            (positions[i], torch.ones_like(positions[i]) * k), -1
                        )
                        for k, i in enumerate(idxes)
                    ],
                    dim=0,
                ),
            }
            t_out, v_out, _ = model(data)
            t_out = t_out[-1]

            if j == 0:
                volume = v_out[-1]

            transforms_diff = []
            for ns in range(len(idxes)):
                idx = data["positions"][:, -1] == ns
                if j > 0 and ns != 2:  # anchor stack
                    transform_diff = transforms_out[ns].compose(t_out[idx].inv())
                    transform_diff = transform_diff.axisangle()
                    mid = transform_diff.shape[0] // 2
                    transforms_diff.append(transform_diff[mid - 3 : mid + 3])
                    continue
                transforms_out.append(t_out[idx])  # new stack
                if j > 0:  # correct stack transformation according to anchor stacks
                    transform_diff = torch.cat(transforms_diff, 0)
                    meanT = transform_diff[:, 3:].mean(0, keepdim=True)
                    meanR = average_rotation(transform_diff[:, :3])
                    transform_diff_mean = torch.cat((meanR, meanT), -1)
                    transforms_out[-1] = RigidTransform(transform_diff_mean).compose(
                        transforms_out[-1]
                    )
    return transforms_out, volume


def run_model_all_stack(transforms, stacks, model, res_s, s_thick, res_r):
    # run models
    device = stacks[0].device
    dtype = stacks[0].dtype
    slice_shape = stacks[0].shape[-2:]

    positions = torch.cat(
        [
            torch.stack(
                (
                    torch.arange(slices.shape[0], dtype=dtype, device=device)
                    - slices.shape[0] // 2,
                    torch.full((slices.shape[0],), i, dtype=dtype, device=device),
                ),
                dim=-1,
            )
            for i, slices in enumerate(stacks)
        ],
        dim=0,
    )

    with torch.no_grad():
        volume_shape = (256, 256, 256)
        data = {
            "psf_rec": get_PSF(
                res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
                device=device,
            ),
            "slice_shape": slice_shape,  # (128, 128)
            "resolution_slice": res_s,
            "resolution_recon": res_r,
            "slice_thickness": s_thick,
            "volume_shape": volume_shape,  # (125, 169, 145),
            "transforms": RigidTransform.cat(transforms).matrix(),
            "stacks": torch.cat(stacks, dim=0),
            "positions": positions,
        }
        t_out, v_out, _ = model(data)
        transforms_out = [t_out[-1][positions[:, -1] == i] for i in range(len(stacks))]
    return transforms_out, v_out[-1]


def parse_data(dataset, res_s):
    stacks = []  # resampled, cropped, normalized
    stacks_ori = []  # resampled
    transforms = []  # cropped, reset (SVoRT input)
    transforms_full = []  # reset, but with original size
    transforms_ori = []  # original
    crop_idx = []  # z

    for data in dataset:
        # resample
        slices = resample(
            data.slices * data.mask,
            (data.resolution_x, data.resolution_y),
            (res_s, res_s),
        )
        slices_ori = slices.clone()
        stacks_ori.append(slices_ori)
        # crop x,y
        s = slices[torch.argmax((slices > 0).sum((1, 2, 3))), 0]
        i1, i2, j1, j2 = 0, s.shape[0] - 1, 0, s.shape[1] - 1
        while s[i1, :].sum() == 0:
            i1 += 1
        while s[i2, :].sum() == 0:
            i2 -= 1
        while s[:, j1].sum() == 0:
            j1 += 1
        while s[:, j2].sum() == 0:
            j2 -= 1
        if (i2 - i1) > 128 or (j2 - j1) > 128:
            logging.warning("ROI in the data is too large for SVoRT")
        pad_margin = 64
        slices = F.pad(
            slices, (pad_margin, pad_margin, pad_margin, pad_margin), "constant", 0
        )
        i = pad_margin + (i1 + i2) // 2
        j = pad_margin + (j1 + j2) // 2
        slices = slices[:, :, i - 64 : i + 64, j - 64 : j + 64]
        # crop z
        nnz = (slices > 0).float().sum((1, 2, 3))
        idx = nnz > 0
        nz = torch.nonzero(idx)
        idx[nz[0, 0] : nz[-1, 0] + 1] = True
        crop_idx.append(idx)
        slices = slices[idx]
        # normalize
        stacks.append(slices / torch.quantile(slices[slices > 0], 0.99))
        # transformation
        transform = data.transformation
        transforms_ori.append(transform)
        transform_full = transform.axisangle().clone()
        transform = transform_full[idx].clone()

        transform_full[:, :-1] = 0
        transform_full[:, 3] = -((j1 + j2) // 2 - slices_ori.shape[-1] / 2) * res_s
        transform_full[:, 4] = -((i1 + i2) // 2 - slices_ori.shape[-2] / 2) * res_s
        transform_full[:, -1] -= transform[:, -1].mean()

        transform[:, :-1] = 0
        transform[:, -1] -= transform[:, -1].mean()

        transforms.append(RigidTransform(transform))
        transforms_full.append(RigidTransform(transform_full))

    return (
        stacks,
        stacks_ori,
        transforms,
        transforms_full,
        transforms_ori,
        crop_idx,
        np.mean([data.thickness for data in dataset]),
    )


def correct_svort(transforms_out, transforms_in, stacks, volume, res_s, s_thick, res_r):
    # correct transorms
    logging.debug("Correcting SVoRT results with stack transformations ...")
    # compute stack transformation
    transforms_stack = []
    for j in range(len(stacks)):
        transform_diff_mean, _ = get_transform_diff_mean(
            transforms_out[j], transforms_in[j]
        )
        transforms_stack.append(transform_diff_mean.compose(transforms_in[j]))

    ncc_stack, weight = simulated_ncc(
        transforms_stack, stacks, volume, res_s, s_thick, res_r
    )
    ncc_svort, _ = simulated_ncc(transforms_out, stacks, volume, res_s, s_thick, res_r)
    # negative NCC (the lower the better)
    logging.debug(
        "%d out of %d slices are replaced with the stack transformation",
        torch.count_nonzero(ncc_svort > ncc_stack).item(),
        ncc_svort.numel(),
    )
    transforms_corrected = []
    idx = 0
    for j in range(len(stacks)):
        ns = stacks[j].shape[0]
        t_out = torch.where(
            (ncc_svort[idx : idx + ns] <= ncc_stack[idx : idx + ns]).reshape(-1, 1, 1),
            transforms_out[j].matrix(),
            transforms_stack[j].matrix(),
        )
        idx += ns
        transforms_corrected.append(RigidTransform(t_out))
    ncc_min = torch.min(ncc_svort, ncc_stack)

    score_svort = compute_score(ncc_min, weight)

    return transforms_corrected, score_svort


def get_transforms_full(transforms_out, transforms_in, transforms_full, crop_idx):
    # full stack
    transforms_svort_full = []
    transforms_stack_full = []
    for j in range(len(transforms_in)):
        transform_diff_mean, transform_diff = get_transform_diff_mean(
            transforms_out[j], transforms_in[j]
        )
        transform_stack_full = transform_diff_mean.compose(transforms_full[j])
        transform_svort_full = transform_stack_full.matrix().clone()
        transform_svort_full[crop_idx[j]] = transform_diff.compose(
            transforms_full[j][crop_idx[j]]
        ).matrix()
        transforms_svort_full.append(RigidTransform(transform_svort_full))
        transforms_stack_full.append(transform_stack_full)

    return transforms_svort_full, transforms_stack_full


def stack_registration(transforms_list, transform_target, stacks, res_s, s_thick):
    # stack registration
    device = transform_target.device

    def t_mean(t):
        return RigidTransform(t.axisangle().mean(0, keepdim=True))

    t_target = t_mean(transform_target)
    ts_in = [
        [t_mean(transform) for transform in transforms]
        for transforms in transforms_list
    ]
    params = {"res_s": res_s, "s_thick": s_thick}
    vvr = VVR(
        num_levels=3,
        num_steps=4,
        step_size=2,
        max_iter=20,
        optimizer={"name": "gd", "momentum": 0.1},
        loss=lambda s, x, y: ncc_loss(x[None], y[None], win=None, reduction="none"),
        auto_grad=False,
    )
    trans_first = False
    ts_registered = []
    for j in range(len(stacks)):
        if j == 0:
            ts_registered.append(t_target)
        else:
            source = stacks[j].squeeze(1)[None, None]
            target = stacks[0].squeeze(1)[None, None]
            ncc_min = float("inf")
            ax_out = None
            for k in range(len(ts_in)):
                ax = (
                    t_target.compose(ts_in[k][0].inv())
                    .compose(ts_in[k][j])
                    .axisangle(trans_first=trans_first)
                )
                ax, ncc = vvr(ax, source, target, params, t_target, trans_first)
                if ncc < ncc_min:
                    ncc_min, ax_out = ncc, ax
            ts_registered.append(RigidTransform(ax_out, trans_first=trans_first))

    t_center = ts_registered[0].axisangle(trans_first=False).clone()
    t_center[..., :3] = 0
    t_center[..., 3:] *= -1
    t_center = RigidTransform(t_center)

    transforms_out = []
    for j in range(len(stacks)):
        n_slice = stacks[j].shape[0]
        t = torch.zeros((n_slice, 6), dtype=torch.float32, device=device)
        t[:, -1] = (
            torch.arange(n_slice, dtype=torch.float32, device=device)
            - (n_slice - 1) / 2
        ) * s_thick
        t = t_center.compose(ts_registered[j]).compose(RigidTransform(t))
        transforms_out.append(t)

    return transforms_out


def reconstruct_from_stacks(transforms, stacks, res_s, s_thick, res_r, n_stack_recon):
    device = stacks[0].device
    size_max = max([max(stacks[j].shape[-2:]) for j in range(len(stacks))])
    stacks_pad = [stack for stack in stacks]
    for j in range(len(stacks)):
        if stacks[j].shape[-1] < size_max or stacks[j].shape[-2] < size_max:
            dx1 = (size_max - stacks[j].shape[-1]) // 2
            dx2 = (size_max - stacks[j].shape[-1]) - dx1
            dy1 = (size_max - stacks[j].shape[-2]) // 2
            dy2 = (size_max - stacks[j].shape[-2]) - dy1
            stacks_pad[j] = F.pad(stacks[j], (dx1, dx2, dy1, dy2))
    # reconstruct volume from stack
    params = {
        "psf": get_PSF(
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
            device=device,
        ),
        "slice_shape": stacks_pad[0].shape[-2:],
        "interp_psf": False,
        "res_s": res_s,
        "res_r": res_r,
        "s_thick": s_thick,
        "volume_shape": (256, 256, 256),
    }
    if n_stack_recon is None:
        n_stack_recon = len(stacks_pad)
    mat = mat_update_resolution(
        RigidTransform.cat([transforms[j] for j in range(n_stack_recon)]).matrix(),
        1,
        res_r,
    )
    ss = torch.cat([stacks_pad[j] for j in range(n_stack_recon)])
    mask_ss = ss > 0
    volume = PSFreconstruction(mat, ss, None, None, params)
    srr = SRR(n_iter=1, use_CG=True)
    volume = srr(mat, ss, volume, params, slices_mask=mask_ss)
    return volume


def simulated_ncc(
    transforms: List[RigidTransform],
    stacks: List[torch.Tensor],
    volume: torch.Tensor,
    res_s: float,
    s_thick: float,
    res_r: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ncc = []
    ncc_weight = []
    psf = get_PSF(
        res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
        device=stacks[0].device,
    )
    for j in range(len(stacks)):
        stack = stacks[j]
        transform = transforms[j]
        stack_mask = stack > 0
        simulated_stack = slice_acquisition(
            mat_update_resolution(transform.matrix(), 1, res_r),
            volume,
            None,
            stack_mask,
            psf,
            stack.shape[-2:],
            res_s / res_r,
            False,
            False,
        )
        ncc_weight.append(stack_mask.sum((1, 2, 3)))
        ncc.append(
            ncc_loss(simulated_stack, stack, stack_mask, win=None, reduction="none")
        )
    ncc_all = torch.cat(ncc)
    ncc_weight_all = torch.cat(ncc_weight).view(ncc_all.shape)
    return ncc_all, ncc_weight_all


def run_svort(dataset, model, svort, vvr, force_vvr):

    res_s = 1.0
    res_r = 0.8

    if svort or vvr:
        (
            stacks_cropped,
            stacks_ori,
            transforms_cropped_reset,
            transforms_ori_reset,
            transforms_ori,
            crop_idx,
            s_thick,
        ) = parse_data(dataset, res_s)

    if svort:
        time_start = time.time()
        if isinstance(model, SVoRT):
            transforms_svort, volume_svort = run_model(
                transforms_cropped_reset, stacks_cropped, model, res_s, s_thick, res_r
            )
        else:
            transforms_svort, volume_svort = run_model_all_stack(
                transforms_cropped_reset, stacks_cropped, model, res_s, s_thick, res_r
            )
        logging.debug("time for running SVoRT: %f s" % (time.time() - time_start))

        time_start = time.time()
        transforms_corrected, score_svort = correct_svort(
            transforms_svort,
            transforms_cropped_reset,
            stacks_cropped,
            volume_svort,
            res_s,
            s_thick,
            res_r,
        )
        logging.debug(
            "time for stack transformation correction: %f s"
            % (time.time() - time_start)
        )

        transforms_svort_full, transforms_stack_full = get_transforms_full(
            transforms_corrected,
            transforms_cropped_reset,
            transforms_ori_reset,
            crop_idx,
        )
    else:
        score_svort = float("-inf")

    if vvr:
        time_start = time.time()
        transforms_vvr = stack_registration(
            [transforms_ori, transforms_stack_full] if svort else [transforms_ori],
            transforms_stack_full[0] if svort else transforms_ori[0],
            stacks_ori,
            res_s,
            s_thick,
        )
        logging.debug("time for stack registration: %f s" % (time.time() - time_start))

        if svort:
            time_start = time.time()
            volume_vvr = reconstruct_from_stacks(
                transforms_vvr,
                stacks_ori,
                res_s,
                s_thick,
                res_r,
                3 if isinstance(model, SVoRT) else None,
            )

            score_vvr = compute_score(
                *simulated_ncc(
                    [t[i] for t, i in zip(transforms_vvr, crop_idx)],
                    [s[i] for s, i in zip(stacks_ori, crop_idx)],
                    volume_vvr,
                    res_s,
                    s_thick,
                    res_r,
                )
            )
            logging.debug(
                "time for evaluating stack registration %f s"
                % (time.time() - time_start)
            )
        else:
            score_vvr = float("inf")
    else:
        score_vvr = float("-inf")

    if svort or vvr:
        if score_svort > float("-inf"):
            logging.info("similarity score for SVoRT = %f", score_svort)
        if score_vvr > float("-inf"):
            logging.info("similarity score for stack registration = %f", score_vvr)
        if score_svort < score_vvr or force_vvr:
            logging.info("use stack transformation")
            transforms_out = transforms_vvr
        else:
            logging.info("use slice transformation")
            transforms_out = transforms_svort_full

        for j in range(len(dataset)):
            dataset[j].transformation = transforms_out[j]

    slices = []
    for stack in dataset:
        idx_nonempty = stack.mask.flatten(1).any(1)
        stack.slices /= torch.quantile(stack.slices[stack.mask], 0.99)
        slices.extend(stack[idx_nonempty])
    dataset = slices

    return dataset


def svort_predict(
    dataset: List[Stack],
    device,
    svort_version: str,
    svort: bool,
    vvr: bool,
    force_vvr: bool,
) -> List[Slice]:
    model: Optional[torch.nn.Module] = None
    if svort:
        if svort_version not in __pretrained_svort:
            raise ValueError("unknown SVoRT version!")
        svort_url = __pretrained_svort[svort_version]
        cp = torch.hub.load_state_dict_from_url(
            url=svort_url,
            model_dir=__checkpoint_dir,
            map_location=device,
            file_name="SVoRT_%s.pt" % svort_version,
        )
        if svort_version == "v1" or "v1." in svort_version:
            model = SVoRT(n_iter=3)
        elif svort_version == "v2" or "v2." in svort_version:
            model = SVoRTv2(n_iter=4)
        else:
            raise ValueError("unknown SVoRT version!")
        model.to(device)
        model.load_state_dict(cp["model"])
        model.eval()
    return run_svort(dataset, model, svort, vvr, force_vvr)
