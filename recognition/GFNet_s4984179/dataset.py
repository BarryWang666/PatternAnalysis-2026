"""COMP3710 | s4984179 | Individual ADNI slice loading.

Input: patient-disjoint CSV metadata and native grayscale JPEG images.
Output: float32 [1,240,256] images, labels and identifiers.
The partition column defines roles; supplied folder names are provenance.
"""

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


PARTITIONS = ("train", "tune", "calibration", "test")


def split_patients(rows, seed=2026):
    """Stratify patients into TRAIN/TUNE/CALIBRATION/TEST at 60/10/10/20.

    Sort patient IDs, then permute AD followed by NC with one PCG64 stream.
    Round the first three allocations; TEST receives the remaining patients.
    Every image inherits its patient's assignment.
    """
    rows = list(rows)
    subjects = {}
    for row in rows:
        subject, label = row["subject_id"], int(row["label"])
        if not subject or label not in (0, 1):
            raise ValueError("Patient IDs and binary labels are required")
        if subjects.setdefault(subject, label) != label:
            raise ValueError("A patient has conflicting diagnosis labels")
    rng = np.random.Generator(np.random.PCG64(seed))
    assigned = {}
    for label in (1, 0):
        ids = sorted(subject for subject in subjects if subjects[subject] == label)
        order = rng.permutation(len(ids))
        counts = [round(len(ids) * fraction) for fraction in (.6, .1, .1)]
        counts.append(len(ids) - sum(counts))
        start = 0
        for partition, count in zip(PARTITIONS, counts):
            for index in order[start:start + count]:
                assigned[ids[int(index)]] = partition
            start += count
    return [{**row, "partition": assigned[row["subject_id"]]} for row in rows]


def build_manifest(data_root, output, seed=2026):
    """Recover patients from course metadata and write a fresh slice-split CSV.

    Read meta_data_with_label.json and AD_NC/{train,test}/{AD,NC} filenames.
    Source labels are AD=2/NC=0; classifier labels are AD=1/NC=0.
    This step reads metadata and filenames, not image pixels.
    """
    root = Path(data_root).resolve()
    metadata = json.loads((root / "meta_data_with_label.json").read_text())
    scans = {}
    for supplied in ("train", "test"):
        for diagnosis, label in (("AD", 1), ("NC", 0)):
            for path in sorted((root / "AD_NC" / supplied / diagnosis).iterdir()):
                if path.suffix.lower() not in (".jpg", ".jpeg"):
                    continue
                scan_id, index = path.stem.rsplit("_", 1)
                record = metadata[scan_id]
                subject = re.search(r"ADNI_(\d+_S_\d+)(?=_|[./]|$)", record["raw"])
                if subject is None or record["label"] != 2 * label:
                    raise ValueError(f"Patient or label mismatch for acquisition {scan_id}")
                scans.setdefault(scan_id, []).append({
                    "relative_image_path": path.relative_to(root).as_posix(),
                    "subject_id": subject[1], "label": label,
                    "scan_id": scan_id, "slice_index": int(index),
                })
    rows = []
    for slices in scans.values():
        for rank, row in enumerate(sorted(slices, key=lambda row: row["slice_index"])):
            rows.append({**row, "slice_rank": rank})
    if not rows:
        raise ValueError("No prepared ADNI slices were found")
    rows = sorted(split_patients(rows, seed), key=lambda row: row["relative_image_path"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "relative_image_path", "subject_id", "label", "partition",
            "scan_id", "slice_index", "slice_rank",
        ))
        writer.writeheader()
        writer.writerows(rows)
    return rows


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
    Native grayscale pixels use [0,1] or [-1,1], without resizing or cropping.
    Optional TRAIN augmentation rotates up to 5 degrees and translates up to
    2% of each axis, using bilinear sampling and black background fill.
    Paths and patient IDs are metadata only; neither is a model input.
    """

    def __init__(self, rows, data_root, partition, *, input_scale="zero_one", augment=False):
        if input_scale not in ("zero_one", "minus_one_one"):
            raise ValueError("Choose zero_one or minus_one_one input scaling")
        if augment and partition != "train":
            raise ValueError("Geometric augmentation is TRAIN-only")
        self.rows = select_partition(rows, partition)
        self.root = Path(data_root).resolve()
        self.partition = partition
        self.input_scale = input_scale
        self.augment = augment

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
            if self.augment:
                angle, dx, dy = (torch.rand(3).tolist())
                image = image.rotate(
                    (2 * angle - 1) * 5, resample=Image.Resampling.BILINEAR,
                    translate=((2 * dx - 1) * 0.02 * 256, (2 * dy - 1) * 0.02 * 240),
                    fillcolor=0,
                )
            pixels = np.array(image, dtype=np.float32) / 255.0
        if self.input_scale == "minus_one_one":
            pixels = 2 * pixels - 1
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the patient-disjoint ADNI slice manifest")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows = build_manifest(args.data_root, args.output, args.seed)
    print(json.dumps({role: sum(row["partition"] == role for row in rows)
                      for role in PARTITIONS}, indent=2))
