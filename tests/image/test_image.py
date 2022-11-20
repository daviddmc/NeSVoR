import unittest
from nesvor.image import load_slices, save_slices, Slice
from nesvor.transform import RigidTransform
import torch
import numpy as np
import os
import shutil


class TestImage(unittest.TestCase):

    tmp_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "tmp_save_load_slices"
    )

    def get_test_data(self):
        ax = [
            [0, 0, 0, 0, 0, 0],
            [np.pi / 2, 0, 0, 1, 2, 3],
            [0, np.pi - 0.01, 0, -1.1, -10, 100.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [-0.1, 0, -0.4, 0.1, 0.5, 0.1],
            [-0.2, 0.2, -0.1, -100, 200, -159],
            [-0.12, -0.01, 0.1, -100, 200, -159],
            [np.pi / 4, np.pi / 4, np.pi / 4, 0.1, 0.1, 0.1],
            [np.pi / 3, -np.pi / 4, np.pi / 5, 100, 200, -300],
        ]
        data = []
        for i, dat in enumerate(ax):
            transformation = RigidTransform(
                torch.tensor([dat], dtype=torch.float32).cuda()
            )
            image = torch.full((1, 128 + i, 256 + i), i, dtype=torch.float32).cuda()
            resolution_x, resolution_y, resolution_z = (
                0.5 + 0.1 * i,
                0.5 + 0.2 * i,
                0.5 + 0.3 * i,
            )
            s = Slice(
                image, None, transformation, resolution_x, resolution_y, resolution_z
            )
            data.append(
                {
                    "slice": s,
                    "image": image,
                    "transformation": transformation,
                    "resolution_x": resolution_x,
                    "resolution_y": resolution_y,
                    "resolution_z": resolution_z,
                }
            )
        return data

    def test_save_load_slices(self):
        data = self.get_test_data()
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)
        os.makedirs(self.tmp_folder)
        save_slices(self.tmp_folder, [dat["slice"] for dat in data])
        slices = load_slices(self.tmp_folder, data[0]["image"].device)
        for i in range(len(data)):
            s = slices[i]
            dat = data[i]
            self.assertAlmostEqual(dat["resolution_x"], s.resolution_x, 3)
            self.assertAlmostEqual(dat["resolution_y"], s.resolution_y, 3)
            self.assertAlmostEqual(dat["resolution_y"], s.resolution_y, 3)
            # print(i)
            # print(dat["transformation"].axisangle().squeeze().cpu().numpy())
            # print(s.transformation.axisangle().squeeze().cpu().numpy())
            self.assert_tensor_close(
                dat["transformation"].axisangle(),
                s.transformation.axisangle(),
                atol=1e-5,
                rtol=1e-3,
            )

            self.assert_tensor_close(dat["image"], s.image)
        shutil.rmtree(self.tmp_folder)

    @staticmethod
    def assert_tensor_close(*args, **kwargs):
        torch.testing.assert_close(*args, **kwargs)
