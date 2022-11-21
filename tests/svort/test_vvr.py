from tests import TestCaseNeSVoR
from nesvor.svort.registration import VVR
from nesvor.transform import RigidTransform
from tests.phantom3d import phantom3d
from nesvor.utils import ncc_loss
import torch


class TestVVR(TestCaseNeSVoR):
    @staticmethod
    def get_vvr_test_data():
        phantom = phantom3d(n=128)
        phantom = torch.tensor(phantom, dtype=torch.float32).cuda()[None, None]
        return phantom

    def test_vvr(self):
        volume = self.get_vvr_test_data()
        vvr = VVR(
            num_levels=3,
            num_steps=8,
            step_size=2,
            max_iter=20,
            optimizer={"name": "gd", "momentum": 0.1},
            loss=lambda s, x, y: ncc_loss(x[None], y[None], win=None, reduction="none"),
            auto_grad=False,
        )

        trans_first = False
        source = volume
        target = volume

        params = {"res_s": 1, "s_thick": 1.5}
        ax = torch.tensor([[0.4, 0.1, -0.6, 20, -50, 100]], dtype=torch.float32).cuda()
        t_target = torch.tensor(
            [[0.4 + 0.05, 0.1 - 0.05, -0.6 + 0.1, 20 + 3, -50 - 2, 100 + 1.5]],
            dtype=torch.float32,
        ).cuda()
        t_target = RigidTransform(t_target, trans_first=trans_first)

        ax_out, _ = vvr(ax, source, target, params, t_target, trans_first)

        self.assert_tensor_close(
            ax_out, t_target.axisangle(trans_first=trans_first), atol=1e-5, rtol=1e-3
        )
