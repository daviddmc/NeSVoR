import unittest
import torch
from scipy.spatial.transform import Rotation
import numpy as np


class TestCaseNeSVoR(unittest.TestCase):
    @staticmethod
    def assert_tensor_close(*args, **kwargs):
        torch.testing.assert_close(*args, **kwargs)

    @staticmethod
    def assert_tensor_equal(*args, **kwargs):
        torch.testing.assert_close(*args, atol=0, rtol=0, **kwargs)

    @staticmethod
    def get_transform_test_data():
        def scipy_axisangle2mat(ax):
            mat = Rotation.from_rotvec(ax.cpu().numpy()[:, :3]).as_matrix()
            mat = torch.tensor(mat, dtype=ax.dtype, device=ax.device)
            mat = torch.cat((mat, ax[:, 3:, None]), -1)
            return mat

        ax = [
            [0, 0, 0, 0, 0, 0],
            [np.pi / 2, 0, 0, 1, 2, 3],
            [0, -np.pi / 2, 0, -1.1, -10, 100.5],
            [0, 0, np.pi - 0.01, 2, 1, 10.5],
            [0, -np.pi + 0.01, 0, 2, 1, 10.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [-0.1, 0, -0.4, 0.1, 0.5, 0.1],
            [-0.2, 0.2, -0.1, -100, 200, -159],
            [-0.12, -0.01, 0.1, -100, 200, -159],
            [np.pi / 4, np.pi / 4, np.pi / 4, 0.1, 0.1, 0.1],
            [np.pi / 3, -np.pi / 4, np.pi / 5, 100, 200, -300],
        ]
        ax = [torch.tensor([dat], dtype=torch.float32).cuda() for dat in ax]
        mat = [scipy_axisangle2mat(dat) for dat in ax]
        return list(zip(ax, mat))
