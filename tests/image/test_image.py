from tests import TestCaseNeSVoR
from nesvor.image import load_slices, save_slices, Slice
from nesvor.transform import RigidTransform
import torch
import os
import shutil


class TestImage(TestCaseNeSVoR):

    tmp_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "tmp_save_load_slices"
    )

    @staticmethod
    def get_image_test_data():
        data = []
        for i, (ax, _) in enumerate(TestCaseNeSVoR.get_transform_test_data()):
            transformation = RigidTransform(ax, trans_first=i % 2 == 1)
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
        data = self.get_image_test_data()
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
