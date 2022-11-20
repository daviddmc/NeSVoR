from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob


root_path = os.path.dirname(__file__)


version: dict = dict()
with open("./nesvor/version.py") as fp:
    exec(fp.read(), version)


def get_long_description():
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description


def get_extensions():
    extensions = [
        CUDAExtension(
            name="nesvor.slice_acq_cuda",
            sources=[
                os.path.join(
                    root_path, "nesvor", "slice_acquisition", "slice_acq_cuda.cpp"
                ),
                os.path.join(
                    root_path, "nesvor", "slice_acquisition", "slice_acq_cuda_kernel.cu"
                ),
            ],
        ),
        CUDAExtension(
            name="nesvor.transform_convert_cuda",
            sources=[
                os.path.join(
                    root_path, "nesvor", "transform", "transform_convert_cuda.cpp"
                ),
                os.path.join(
                    root_path, "nesvor", "transform", "transform_convert_cuda_kernel.cu"
                ),
            ],
        ),
    ]
    return extensions


def get_package_data():
    ext_src = []
    for ext in ["cpp", "cu", "h", "cuh"]:
        ext_src.extend(
            glob.glob(os.path.join("nesvor", "**", f"*.{ext}"), recursive=True)
        )
    return {"nesvor": ["py.typed"] + [os.path.join("..", path) for path in ext_src]}


def get_entry_points():
    entry_points = {
        "console_scripts": ["nesvor=nesvor.cli.main:main"],
    }
    return entry_points


setup(
    name="nesvor",
    packages=find_packages(exclude=("tests",)),
    version=version["__version__"],
    description="NeSVoR: toolkit for neural slice-to-volume reconstruction",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/daviddmc/NeSVoR",
    author="Junshen Xu",
    author_email="junshen@mit.edu",
    license="MIT",
    zip_safe=False,
    entry_points=get_entry_points(),
    ext_modules=get_extensions(),
    package_data=get_package_data(),
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)
