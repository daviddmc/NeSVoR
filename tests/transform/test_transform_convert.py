from nesvor.transform import (
    axisangle2mat,
    mat2axisangle,
    mat2point,
    point2mat,
    mat2euler,
    euler2mat,
)
from tests import TestCaseNeSVoR


class TestTransformConvert(TestCaseNeSVoR):
    def test_axisangle2mat(self):
        for ax, mat in self.get_transform_test_data():
            mat_ = axisangle2mat(ax)
            self.assert_tensor_close(mat_, mat)

    def test_mat2axisangle(self):
        for ax, mat in self.get_transform_test_data():
            ax_ = mat2axisangle(mat)
            self.assert_tensor_close(ax_, ax)

    def test_mat2point_point2mat(self):
        for i, (_, mat) in enumerate(self.get_transform_test_data()):
            p = mat2point(mat, 128 + 2 * i, 128 + 4 * i, 0.5 + 0.1 * i)
            mat_ = point2mat(p)
            self.assert_tensor_close(mat_, mat)

    def test_mat2euler_euler2mat(self):
        for i, (_, mat) in enumerate(self.get_transform_test_data()):
            euler = mat2euler(mat)
            mat_ = euler2mat(euler)
            self.assert_tensor_close(mat_, mat)
