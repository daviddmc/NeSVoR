import unittest
from nesvor.transform import axisangle2mat, mat2axisangle
import torch
from scipy.spatial.transform import Rotation
import numpy as np


class TestTransformConvert(unittest.TestCase):
    @staticmethod
    def scipy_axisangle2mat(ax):
        mat = Rotation.from_rotvec(ax.cpu().numpy()[:, :3]).as_matrix()
        mat = torch.tensor(mat, dtype=ax.dtype, device=ax.device)
        mat = torch.cat((mat, ax[:, 3:, None]), -1)
        return mat

    def get_test_data(self):
        ax = [
            [0, 0, 0, 0, 0, 0],
            [np.pi / 2, 0, 0, 1, 2, 3],
            [0, -np.pi / 2, 0, -1.1, -10, 100.5],
            [0, 0, np.pi - 0.01, 2, 1, 10.5],
            [0, -np.pi + 0.01, 0, 2, 1, 10.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [-0.1, 0, -0.4, 0.1, 0.5, 0.1],
        ]
        ax = [torch.tensor([dat], dtype=torch.float32).cuda() for dat in ax]
        mat = [self.scipy_axisangle2mat(dat) for dat in ax]
        return zip(ax, mat)

    def test_axisangle2mat(self):
        for ax, mat in self.get_test_data():
            mat_ = axisangle2mat(ax)
            self.assert_tensor_close(mat_, mat)

    def test_mat2axisangle(self):
        for ax, mat in self.get_test_data():
            ax_ = mat2axisangle(mat)
            self.assert_tensor_close(ax_, ax)

    @staticmethod
    def assert_tensor_close(*args, **kwargs):
        torch.testing.assert_close(*args, **kwargs)
