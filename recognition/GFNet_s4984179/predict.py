"""COMP3710 | s4984179 | Individual slice inference and reliability analysis.

Input: selected weights and image metadata, or saved prediction records.
Output: per-slice probabilities, classification/confidence metrics and plots.
Metrics operate on slices; patient IDs are metadata rather than predictions.
"""


import argparse


from collections import defaultdict


from fractions import Fraction


import math


import json


from pathlib import Path


import numpy as np


import torch


from dataset import ADNISlices, make_loader, read_manifest


from train import a100_runtime, build_model, file_digest, probabilities, slice_metrics, write_json, scan_summary, weighted_probability_metrics


def profile_scan(model, images):
    """Measure 50 warmups/200 CUDA-event repetitions for a batch of 20 slices.

    Includes forward, softmax and probability mean; excludes decode and transfer.
    These scan-batch timings are not single-slice or end-to-end latency.
    """
    if images.shape != (20,1,240,256) or images.device.type != 'cuda':
        raise ValueError('Expected an allocated CUDA batch of twenty native slices')
    model.eval();timings=[]
    with torch.inference_mode():
        for _ in range(50): torch.softmax(model(images),dim=1)[:,1].mean()
        torch.cuda.synchronize(images.device)
        torch.cuda.reset_peak_memory_stats(images.device)
        for _ in range(200):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();torch.softmax(model(images),dim=1)[:,1].mean();end.record();end.synchronize()
            timings.append(start.elapsed_time(end))
    return dict(timings_ms=timings,median_ms=float(np.median(timings)),
                p95_ms=float(np.percentile(timings,95,method='linear')),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(images.device))
