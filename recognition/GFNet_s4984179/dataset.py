"""COMP3710 | s4984179 | Individual ADNI slice loading.

Input: patient-disjoint CSV metadata and native grayscale JPEG images.
Output: float32 [1,240,256] images, labels and identifiers.
The partition column defines roles; supplied folder names are provenance.
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


PARTITIONS = ("train", "tune", "calibration", "test")


def read_manifest(path):
    """Read a CSV with relative_image_path, subject_id, label and partition.

    Return rows sorted by path, with NC=0/AD=1 integer labels. Reject duplicate
    images and patients assigned to multiple partitions before any image I/O.
    """
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"relative_image_path", "subject_id", "label", "partition"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("Manifest needs relative_image_path, subject_id, label and partition")

    images, patients = set(), {}
    for row in rows:
        relative = Path(row["relative_image_path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Image paths must stay inside the data directory")
        if row["label"] not in ("0", "1") or row["partition"] not in PARTITIONS:
            raise ValueError("Expected labels NC=0 / AD=1 and a declared partition")
        if not row["subject_id"] or row["relative_image_path"] in images:
            raise ValueError("Missing patient ID or duplicate image")
        if patients.setdefault(row["subject_id"], row["partition"]) != row["partition"]:
            raise ValueError("Patient appears in more than one partition")
        images.add(row["relative_image_path"])
        row["label"] = int(row["label"])
    return sorted(rows, key=lambda row: row["relative_image_path"])


def select_partition(rows, partition):
    """Keep all slices assigned to one manifest partition, in input order."""
    if partition not in PARTITIONS:
        raise ValueError("Choose train, tune, calibration or test explicitly")
    selected = [row for row in rows if row["partition"] == partition]
    if not selected:
        raise ValueError(f"No images in {partition}")
    return selected


class ADNISlices(Dataset):
    """Load one declared partition from manifest rows under data_root.

    Each item contains image [1,240,256], integer label, path and subject_id.
    Native grayscale pixels are scaled to [0,1], without resizing or cropping.
    Paths and patient IDs are metadata only; neither is a model input.
    """

    def __init__(self, rows, data_root, partition):
        self.rows = select_partition(rows, partition)
        self.root = Path(data_root).resolve()
        self.partition = partition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["relative_image_path"]).resolve()
        if self.root not in path.parents:
            raise ValueError("Image resolves outside the data directory")
        with Image.open(path) as image:
            if image.format != "JPEG" or image.mode != "L" or image.size != (256, 240):
                raise ValueError(f"Expected grayscale 256x240 JPEG: {path}")
            pixels = np.array(image, dtype=np.float32) / 255.0
        return {
            "image": torch.from_numpy(pixels[None]),
            "label": int(row["label"]),
            "path": row["relative_image_path"],
            "subject_id": row["subject_id"],
        }


def make_loader(data, batch_size=64, seed=3710, epoch=0):
    """Shuffle TRAIN without replacement; retain every slice and final batch.

    A private generator seeded by seed + epoch supports recreating the loader
    for each epoch without depending on the model's random draws.
    """
    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(
        data, batch_size=batch_size, shuffle=data.partition == "train",
        drop_last=False, num_workers=0, generator=generator,
    )
