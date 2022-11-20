from nesvor.transform import RigidTransform
from tests import TestCaseNeSVoR
import torch


class TestTransform(TestCaseNeSVoR):
    def test_compose_inv(self):
        zeros = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float32).cuda()
        data = self.get_transform_test_data()
        for i in range(len(data)):
            ax_a, mat_a = data[i]
            ax_b, mat_b = data[-i - 1]
            ab = RigidTransform(ax_a, trans_first=i % 2 == 0).compose(
                RigidTransform(mat_b, trans_first=i % 2 == 1)
            )
            inv_b_inv_a = (
                RigidTransform(ax_b, trans_first=i % 2 == 1)
                .inv()
                .compose(RigidTransform(mat_a, trans_first=i % 2 == 0).inv())
            )
            self.assert_tensor_close(
                ab.compose(inv_b_inv_a).axisangle(), zeros, atol=2e-5, rtol=1e-3
            )
