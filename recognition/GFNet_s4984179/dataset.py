"""COMP3710 | s4984179 | Individual ADNI slice loading.

Input: patient-disjoint CSV metadata and native grayscale JPEG images.
Output: float32 [1,240,256] images, labels and identifiers.
The partition column defines roles; supplied folder names are provenance.
"""


import csv


import re


from collections import defaultdict


from pathlib import Path


import numpy as np


from PIL import Image


import torch


from torch.utils.data import DataLoader, Dataset


PARTITIONS = ("train", "tune", "calibration", "test")


def recover_subject(raw_path):
    """Recover the ADNI patient token rather than using the acquisition prefix."""
    matches = re.findall(r"ADNI_(\d+_S_\d+)(?=_|\.|/|$)", raw_path)
    if len(matches) != 1:
        raise ValueError("Expected one ADNI subject identifier")
    return matches[0]
