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
    return sorted(rows, key=lambda row: (row["subject_id"], row["scan_id"],
                                          int(row["slice_index"]), row["relative_image_path"]))


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


def recover_subject(raw_path):
    """Recover the ADNI patient token rather than using the acquisition prefix."""
    matches = re.findall(r"ADNI_(\d+_S_\d+)(?=_|\.|/|$)", raw_path)
    if len(matches) != 1:
        raise ValueError("Expected one ADNI subject identifier")
    return matches[0]


def assign_patients(subjects):
    """Assign sorted diagnosis strata with one PCG64(2026) stream.

    The fixed production counts are AD 133/22/22/44 and NC 275/46/46/92.
    Return the patient-to-role mapping; image rows inherit the patient's role.
    """
    allocations = {"AD": (133,22,22,44), "NC": (275,46,46,92)}
    rng = np.random.Generator(np.random.PCG64(2026))
    result = {}
    for diagnosis in ("AD", "NC"):
        ids = sorted(r["subject_id"] for r in subjects if r["diagnosis"] == diagnosis)
        if len(ids) != sum(allocations[diagnosis]):
            raise ValueError("Unexpected diagnosis population")
        permutation = rng.permutation(len(ids))
        start = 0
        for role, count in zip(PARTITIONS, allocations[diagnosis]):
            for index in permutation[start:start+count]:
                result[ids[int(index)]] = role
            start += count
    return result


class SubjectScanSampler:
    """Choose one scan per TRAIN patient, retain its slices, then shuffle.

    Separate PCG64 streams fix visit choice and slice order per seed/epoch.
    The fixed manifest has twenty slices per scan: 408 patients give 8,160
    presentations. All TUNE slices remain available for evaluation.
    """
    def __init__(self, data, seed=3710, epoch=0):
        self.seed, self.epoch = seed, epoch
        groups = defaultdict(lambda: defaultdict(list))
        for index,row in enumerate(data.rows):
            groups[row["subject_id"]][row["scan_id"]].append(index)
        self.groups = tuple(tuple(tuple(sorted(indices, key=lambda i: (
            int(data.rows[i]["slice_index"]), data.rows[i]["relative_image_path"])))
            for _,indices in sorted(scans.items())) for _,scans in sorted(groups.items()))
    def __iter__(self):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x5343414E,self.epoch,self.seed])))
        selected = [i for scans in self.groups for i in scans[int(rng.integers(len(scans)))]]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x534C4943,self.epoch,self.seed])))
        return iter(selected[int(i)] for i in rng.permutation(len(selected)))
    def __len__(self):
        return sum(len(scans[0]) for scans in self.groups)


def make_loader(data, batch_size=64, seed=3710, epoch=0):
    """Use patient/scan sampling for TRAIN and complete ordered evaluation."""
    worker_seed = int(np.random.SeedSequence([0x4C4F4144,epoch,seed]).generate_state(1,dtype=np.uint64)[0])
    generator = torch.Generator().manual_seed(worker_seed)
    sampler = SubjectScanSampler(data,seed,epoch) if data.partition == "train" else None
    return DataLoader(data,batch_size=batch_size,sampler=sampler,shuffle=False,
                      drop_last=False,num_workers=0,generator=generator)
