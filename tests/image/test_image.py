from tests import TestCaseNeSVoR
from nesvor.image import load_slices, save_slices, Slice, Volume, load_volume
from nesvor.transform import RigidTransform
import torch
import os
import shutil


class TestImage(TestCaseNeSVoR):
    @staticmethod
    def get_image_test_data(is_volume=False):
        data = []
        for i, (ax, _) in enumerate(TestCaseNeSVoR.get_transform_test_data()):
            transformation = RigidTransform(ax, trans_first=i % 2 == 1)
            image = torch.full(
                ((128 - i) if is_volume else 1, 128 + i, 256 + i),
                i,
                dtype=torch.float32,
            ).cuda()
            resolution_x, resolution_y, resolution_z = (
                0.5 + 0.1 * i,
                0.5 + 0.2 * i,
                0.5 + 0.3 * i,
            )
            C = Volume if is_volume else Slice
            s = C(image, None, transformation, resolution_x, resolution_y, resolution_z)
            data.append(
                {
                    "object": s,
                    "image": image,
                    "transformation": transformation,
                    "resolution_x": resolution_x,
                    "resolution_y": resolution_y,
                    "resolution_z": resolution_z,
                }
            )
        return data

    def test_save_load_slices(self):
        tmp_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "tmp_save_load_slices"
        )
        data = self.get_image_test_data(is_volume=False)
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)
        save_slices(tmp_folder, [dat["object"] for dat in data])
        slices = load_slices(tmp_folder, data[0]["image"].device)
        for i in range(len(data)):
            s = slices[i]
            dat = data[i]
            self.assertAlmostEqual(dat["resolution_x"], s.resolution_x, 3)
            self.assertAlmostEqual(dat["resolution_y"], s.resolution_y, 3)
            self.assertAlmostEqual(dat["resolution_z"], s.resolution_z, 3)
            self.assert_tensor_close(
                dat["transformation"].axisangle(),
                s.transformation.axisangle(),
                atol=1e-5,
                rtol=1e-3,
            )
            self.assert_tensor_close(dat["image"], s.image)
        shutil.rmtree(tmp_folder)

    def test_save_load_volume(self):
        tmp_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "tmp_save_load_volume"
        )
        data = self.get_image_test_data(is_volume=True)
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)
        for i in range(len(data)):
            v = data[i]["object"]
            v.save(os.path.join(tmp_folder, "%d.nii.gz" % i))
            v_ = load_volume(
                os.path.join(tmp_folder, "%d.nii.gz" % i),
                device=data[0]["image"].device,
            )
            self.assertAlmostEqual(v_.resolution_x, v.resolution_x, 3)
            self.assertAlmostEqual(v_.resolution_y, v.resolution_y, 3)
            self.assertAlmostEqual(v_.resolution_z, v.resolution_z, 3)
            self.assert_tensor_close(
                v_.transformation.axisangle(),
                v.transformation.axisangle(),
                atol=1e-5,
                rtol=1e-3,
            )
            self.assert_tensor_close(v_.image, v.image)
        shutil.rmtree(tmp_folder)
