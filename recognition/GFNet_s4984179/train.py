"""COMP3710 | s4984179 | Patient-sampled training and weighted scan TUNE selection.

Input: slice manifest, image root and CNN/GFNet configuration.
Output: selected-best/latest weights, epoch metrics and a run summary.
Usage: python train.py --help; production runs require an allocated A100.
"""


import argparse


from collections import defaultdict


from dataclasses import asdict, dataclass


import hashlib


import json


import math


import os


import platform


import random


from pathlib import Path


import time


import numpy as np


import torch


from torch.nn import functional as F


from dataset import ADNISlices, make_loader, read_manifest


def probabilities(margins):
    """Stable AD probabilities from AD-minus-NC logits."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(margins, dtype=np.float64)))


def weighted_probability_metrics(labels, probability, weights=None):
    """Binary probability measures with explicit weight mass and clipped NLL."""
    y,p = np.asarray(labels,dtype=int),np.asarray(probability,dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    if not len(y) or p.shape != y.shape or w.shape != y.shape or np.any(w<0) or not np.isfinite(p).all():
        raise ValueError("Need aligned, finite probabilities and nonnegative weights")
    total = math.fsum(w)
    if total <= 0 or np.any((p<0)|(p>1)):
        raise ValueError("Need positive weight mass and probabilities in [0,1]")
    q = np.clip(p,1e-7,1-1e-7)
    correct = (p>=.5)==y
    return dict(n=len(y),weight_mass=total,
        accuracy=math.fsum(w[correct])/total,
        nll=math.fsum(w*(-y*np.log(q)-(1-y)*np.log1p(-q)))/total,
        brier=math.fsum(w*(p-y)**2)/total)


def scan_summary(rows, records):
    """Average slice probabilities per scan and give each patient weight one.

    Each scan receives 1/V where V is that patient's number of scans. The
    resulting weighted scan NLL is the V1 checkpoint-selection objective.
    """
    by_path = {r["path"]:float(probabilities(r["margin"])) for r in records}
    if len(by_path)!=len(records) or set(by_path)!={r["relative_image_path"] for r in rows}:
        raise ValueError("Need one probability for every declared slice")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["subject_id"],row["scan_id"])].append(row)
    visits = defaultdict(int)
    for subject,_ in groups: visits[subject]+=1
    scans=[]
    for (subject,scan),members in sorted(groups.items()):
        p=math.fsum(by_path[r["relative_image_path"]] for r in members)/len(members)
        scans.append(dict(subject_id=subject,scan_id=scan,label=int(members[0]["label"]),
                          probability_ad=p,weight=1/visits[subject],visits=visits[subject]))
    report=weighted_probability_metrics([r["label"] for r in scans],
        [r["probability_ad"] for r in scans],[r["weight"] for r in scans])
    return dict(scans=scans,weighted_scan_nll=report["nll"],weighted_scan_metrics=report,
                scan_count=len(scans),subject_count=len(visits))
