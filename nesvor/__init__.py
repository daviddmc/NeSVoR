import os
from .version import *

__base_dir = os.path.dirname(__file__)
__checkpoint_dir = os.path.join(__base_dir, "checkpoints")
__pretrained_svort = {
    "v1": "https://zenodo.org/record/7486938/files/checkpoint.pt?download=1",
    "v2": "https://zenodo.org/record/7486938/files/checkpoint_v2.pt?download=1",
}
