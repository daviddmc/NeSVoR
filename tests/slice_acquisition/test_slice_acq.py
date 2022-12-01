from tests import TestCaseNeSVoR
from nesvor.transform import RigidTransform, mat_update_resolution
from nesvor.slice_acquisition import slice_acquisition
from nesvor.utils import get_PSF
from nesvor.svort.srr import SRR
from tests.phantom3d import phantom3d
import torch
import numpy as np


class TestSliceAcq(TestCaseNeSVoR):
    @staticmethod
    def get_cg_recon_test_data():
        vs = 32
        gap = s_thick = 3
        res = 1
        res_s = 1.5
        n_slice = int((np.sqrt(3) * vs) / gap) + 4
        ss = int((np.sqrt(3) * vs) / res_s) + 4

        volume = phantom3d(n=vs)
        volume = torch.tensor(volume, dtype=torch.float32).cuda()[None, None]
        psf = get_PSF(res_ratio=(res_s / res, res_s / res, s_thick / res)).cuda()
        angles = [
            [0, 0, 0],
            [np.pi / 2, 0, 0],
            [0, np.pi / 2, 0],
            [0, 0, np.pi / 2],
            [np.pi / 4, np.pi / 4, 0],
            [0, np.pi / 4, np.pi / 4],
            [np.pi / 4, 0, np.pi / 4],
            [np.pi / 3, np.pi / 3, 0],
            [0, np.pi / 3, np.pi / 3],
            [np.pi / 3, 0, np.pi / 3],
            [2 * np.pi / 3, 2 * np.pi / 3, 0],
            [0, 2 * np.pi / 3, 2 * np.pi / 3],
            [2 * np.pi / 3, 0, 2 * np.pi / 3],
            [np.pi / 5, np.pi / 5, 0],
            [0, np.pi / 5, np.pi / 5],
            [np.pi / 5, 0, np.pi / 5],
        ]

        stacks = []
        transforms = []
        for i in range(len(angles)):
            angle = (
                torch.tensor([angles[i]], dtype=torch.float32)
                .cuda()
                .expand(n_slice, -1)
            )
            tz = (
                torch.arange(0, n_slice, dtype=torch.float32).cuda()
                - (n_slice - 1) / 2.0
            ) * gap
            tx = ty = torch.ones_like(tz) * 0.5
            t = torch.stack((tx, ty, tz), -1)
            transform = RigidTransform(torch.cat((angle, t), -1), trans_first=True)
            # sample slices
            mat = mat_update_resolution(transform.matrix(), 1, res)
            slices = slice_acquisition(
                mat, volume, None, None, psf, (ss, ss), res_s / res, False, False
            )
            stacks.append(slices)
            transforms.append(transform)
        params = {
            "psf": psf,
            "slice_shape": (ss, ss),
            "res_s": res_s,
            "res_r": res,
            "interp_psf": False,
            "volume_shape": (vs, vs, vs),
        }

        return torch.cat(stacks, 0), RigidTransform.cat(transforms), volume, params

    def test_cg_recon(self):
        slices, transforms, volume, params = self.get_cg_recon_test_data()
        srr = SRR(n_iter=20, use_CG=True, tol=1e-8)
        theta = mat_update_resolution(transforms.matrix(), 1, params["res_r"])
        volume_ = srr(theta, slices, volume, params)
        self.assert_tensor_close(volume_, volume, atol=3e-5, rtol=1e-5)
