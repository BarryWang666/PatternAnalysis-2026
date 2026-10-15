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


def temperature_scale(probability, temperature):
    """Apply positive temperature to clipped scan logits, without fitting."""
    p=np.clip(np.asarray(probability,dtype=float),1e-7,1-1e-7)
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be finite and positive')
    return probabilities(np.log(p/(1-p))/temperature)


def selective_metrics(labels, probability, weights, threshold):
    """Measure fixed-policy acceptance; preserve full-population AD support."""
    y,p,w=np.asarray(labels),np.asarray(probability),np.asarray(weights,dtype=float)
    accepted=np.maximum(p,1-p)>=threshold
    correct=(p>=.5)==y
    total=math.fsum(w)
    mass=math.fsum(w[accepted])
    risk=math.fsum(w[accepted & ~correct])/mass if mass else None
    ad=math.fsum(w[y==1])
    return dict(coverage=mass/total,rejected_fraction=math.fsum(w[~accepted])/total,
        accepted_count=int(accepted.sum()),risk=risk,
        accepted_false_negatives=int(np.sum(accepted & (y==1) & (p<.5))),
        accepted_fn_fraction_all_AD=math.fsum(w[accepted & (y==1) & (p<.5)])/ad if ad else None)


def fit_scan_policy(rows, records):
    """Fit scan temperature and a >=80% weighted-coverage threshold on CALIBRATION."""
    if not rows or any(r['partition']!='calibration' for r in rows):
        raise ValueError('Fit only the CALIBRATION population')
    scans=scan_summary(rows,records)['scans']
    y=[r['label'] for r in scans];p=[r['probability_ad'] for r in scans];w=[r['weight'] for r in scans]
    temperatures=np.unique(np.r_[np.geomspace(.05,20.,201),1.])
    candidates=[(weighted_probability_metrics(y,temperature_scale(p,float(t)),w)['nll'],abs(float(t)-1),float(t))
                for t in temperatures]
    nll,_,temperature=min(candidates)
    q=temperature_scale(p,temperature);confidence=np.maximum(q,1-q)
    exact=[Fraction(1,r['visits']) for r in scans]
    for threshold in np.unique(confidence)[::-1]:
        mass=sum((v for v,c in zip(exact,confidence) if c>=threshold),Fraction(0))
        if mass/sum(exact,Fraction(0))>=Fraction(4,5): break
    return dict(temperature=temperature,threshold=float(threshold),fit_partition='calibration',
                target_coverage=.8,nll=nll,slice_count=len(rows),
                calibration=selective_metrics(y,q,w,float(threshold)))


def apply_scan_policy(policy, rows, records):
    """Apply an already fitted scan policy; evaluation never changes its parameters."""
    scans=scan_summary(rows,records)['scans']
    y=[r['label'] for r in scans];w=[r['weight'] for r in scans]
    p=temperature_scale([r['probability_ad'] for r in scans],policy['temperature'])
    return dict(metrics=weighted_probability_metrics(y,p,w),
                selective=selective_metrics(y,p,w,policy['threshold']))


def risk_coverage_curve(labels, probability, weights=None):
    """Return descriptive points admitting whole tied-confidence groups."""
    y,p=np.asarray(labels),np.asarray(probability)
    w=np.ones(len(y)) if weights is None else np.asarray(weights)
    confidence=np.maximum(p,1-p)
    result=[dict(coverage=0.,risk=None,threshold=None)]
    for threshold in np.unique(confidence)[::-1]:
        result.append(dict(threshold=float(threshold),**selective_metrics(y,p,w,float(threshold))))
    return result
