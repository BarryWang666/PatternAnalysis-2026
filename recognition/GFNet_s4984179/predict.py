"""COMP3710 | s4984179 | Individual slice inference and reliability analysis.

Input: selected weights and image metadata, or saved prediction records.
Output: per-slice probabilities, classification/confidence metrics and plots.
Metrics operate on slices; patient IDs are metadata rather than predictions.
"""

import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from dataset import ADNISlices, make_loader, read_manifest
from train import a100_runtime, build_model, file_digest, probabilities, slice_metrics, write_json


def load_selected_model(checkpoint, device="cpu"):
    """Load best.pt on device and return (evaluation model, checkpoint metadata).

    The configuration chooses CNN/GFNet; training optimizer and RNG are not
    restored. latest.pt is reserved for resume and is rejected here.
    """
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "slice-best-v1":
        raise ValueError("Use the selected best.pt checkpoint, not latest.pt")
    config = saved["identity"]["config"]
    model = build_model(config["model"], config.get("gfnet_embedding", "patch"),
                        config.get("gfnet_depth", 12), config.get("pyramid_depths", (2, 2, 2)),
                        config.get("gfnet_mlp_ratio", 4))
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    metadata = {key: value for key, value in saved.items() if key != "model"}
    return model, metadata


def predict_slices(model, loader, device="cpu", *, profile=None):
    """Return one record per loader image, including the final partial batch.

    Records contain path, subject ID, label, class, margin, AD probability and
    confidence. Evaluation/inference mode prevents parameter or gradient updates.
    """
    model.to(device).eval()
    records = []
    cuda = torch.device(device).type == "cuda"
    forward_seconds, batches = 0.0, 0
    if profile is not None and cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            if profile is not None and cuda:
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            logits = model(images)
            if profile is not None and cuda:
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - before
            batches += 1
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite prediction logits")
            margins = (logits[:, 1] - logits[:, 0]).cpu().tolist()
            for path, subject, label, margin, probability, confidence in zip(
                batch["path"], batch["subject_id"], batch["label"].tolist(),
                margins, probabilities(margins), probabilities(np.abs(margins)),
            ):
                records.append({
                    "path": path, "subject_id": subject, "label": int(label),
                    "prediction": int(margin >= 0), "margin": margin,
                    "probability_ad": float(probability),
                    "confidence": float(confidence),
                })
    if not records or len(records) != len(loader.dataset):
        raise ValueError("Prediction must cover every slice")
    if profile is not None:
        profile.update(n=len(records), batches=batches, wall_seconds=time.perf_counter() - started,
                       forward_seconds=forward_seconds,
                       forward_ms_per_slice=1000 * forward_seconds / len(records),
                       forward_ms_per_batch=1000 * forward_seconds / batches,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None)
    return records


def auroc(labels, scores):
    """Binary AUROC with average ranks for tied AD scores."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def classification_report(records):
    """Report classification from records containing label and raw margin.

    Include every supplied slice. Return confusion, class precision/recall/F1,
    macro-F1, AD AUROC and training-compatible diagnostics; undefined values are None.
    """
    records = list(records)
    labels = [row["label"] for row in records]
    margins = [row["margin"] for row in records]
    result = slice_metrics(labels, margins)
    matrix = result["confusion"]
    classes = {}
    for label, name in enumerate(("NC", "AD")):
        tp = matrix[label][label]
        support = sum(matrix[label])
        chosen = sum(row[label] for row in matrix)
        classes[name] = {
            "support": support, "predicted": chosen,
            "precision": tp / chosen if chosen else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + chosen) if support + chosen else None,
        }
    f1 = [value["f1"] for value in classes.values()]
    result.update({
        "per_class": classes,
        "macro_f1": float(np.mean(f1)) if all(value is not None for value in f1) else None,
        "auroc": auroc(labels, margins),
    })
    return result


def confidence_report(records):
    """Add raw confidence reliability and error-discrimination diagnostics."""
    records = list(records)
    result = classification_report(records)
    labels = np.asarray([row["label"] for row in records])
    margins = np.asarray([row["margin"] for row in records], dtype=np.float64)
    probability = probabilities(margins)
    confidence = probabilities(np.abs(margins))
    correct = (margins >= 0) == labels
    high_count = result["high_confidence_count"]
    correlation = None
    if np.ptp(confidence) > 0 and correct.any() and (~correct).any():
        correlation = float(np.corrcoef(confidence, correct.astype(float))[0, 1])
    result.update({
        "brier": float(np.mean((probability - labels) ** 2)),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "high_confidence_error_rate": result["high_confidence_errors"] / high_count if high_count else None,
        "error_detection_auroc": auroc((~correct).astype(int), -np.abs(margins)),
        "confidence_correctness_correlation": correlation,
    })
    return result


def reliability_report(records, n_bins=10):
    """Add equal-width bins on [0.5,1] and sample-weighted ECE to raw metrics.

    Input records contain label and margin. Each output bin has count, mean
    confidence and accuracy; empty means are None, and the final bin includes 1.
    """
    if n_bins < 1:
        raise ValueError("Use at least one confidence bin")
    records = list(records)
    result = confidence_report(records)
    margins = np.asarray([row["margin"] for row in records], dtype=np.float64)
    labels = np.asarray([row["label"] for row in records])
    confidence = probabilities(np.abs(margins))
    correct = (margins >= 0) == labels
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins, ece = [], 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (confidence >= lower) & (confidence <= upper if index == n_bins - 1 else confidence < upper)
        count = int(inside.sum())
        accuracy = float(correct[inside].mean()) if count else None
        mean_confidence = float(confidence[inside].mean()) if count else None
        if count:
            ece += count / len(records) * abs(accuracy - mean_confidence)
        bins.append({"lower": float(lower), "upper": float(upper), "count": count,
                     "accuracy": accuracy, "confidence": mean_confidence})
    result.update(ece=float(ece), reliability_bins=bins)
    return result


def scale_confidence(records, temperature):
    """Return per-slice probabilities from margins/T, preserving raw margins."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    records = list(records)
    margins = np.asarray([row["margin"] for row in records]) / temperature
    return [{**row, "raw_margin": row["margin"], "margin": float(margin),
             "prediction": int(margin >= 0), "probability_ad": float(probability),
             "confidence": float(confidence)}
            for row, margin, probability, confidence in zip(
                records, margins, probabilities(margins), probabilities(np.abs(margins)))]


def fit_calibration(records):
    """Fit slice NLL temperature and an at-least-80% coverage threshold.

    Call with CALIBRATION records only. Search log T on the fixed [.05,20]
    interval; compare endpoints and T=1. Retain complete score ties. The
    absolute scaled margin is the operative score, avoiding sigmoid saturation.
    """
    records = list(records)
    labels = np.asarray([row["label"] for row in records])
    margins = np.asarray([row["margin"] for row in records], dtype=np.float64)
    raw = slice_metrics(labels, margins)

    def nll(temperature):
        z = margins / temperature
        return float(np.mean(np.logaddexp(0, z) - labels * z))

    lower, upper = math.log(.05), math.log(20)
    fraction = (math.sqrt(5) - 1) / 2
    left, right = upper - fraction * (upper - lower), lower + fraction * (upper - lower)
    a, b = nll(math.exp(left)), nll(math.exp(right))
    for _ in range(80):
        if a <= b:
            upper, right, b = right, left, a
            left = upper - fraction * (upper - lower)
            a = nll(math.exp(left))
        else:
            lower, left, a = left, right, b
            right = lower + fraction * (upper - lower)
            b = nll(math.exp(right))
    candidates = (1.0, .05, 20.0, math.exp((lower + upper) / 2))
    temperature = min(candidates, key=lambda value: (nll(value), abs(math.log(value))))
    scores = np.abs(margins / temperature)
    retain = (4 * len(records) + 4) // 5
    threshold = float(np.sort(scores)[-retain])
    return {"temperature": temperature, "temperature_bounds": [.05, 20.0],
            "temperature_at_bound": temperature in (.05, 20.0),
            "target_coverage": .8, "score_threshold": threshold,
            "confidence_threshold": float(probabilities(threshold)),
            "calibration_n": len(records), "calibration_coverage": float(np.mean(scores >= threshold)),
            "raw_nll": raw["nll"], "calibrated_nll": nll(temperature)}


def selective_report(records, score_threshold):
    """Apply a fixed absolute-margin threshold; do not refit on evaluation."""
    records = list(records)
    labels = np.asarray([row["label"] for row in records])
    margins = np.asarray([row["margin"] for row in records])
    accepted = np.abs(margins) >= score_threshold
    wrong = (margins >= 0) != labels
    n = int(accepted.sum())
    errors = int((accepted & wrong).sum())
    false_negatives = int((accepted & (labels == 1) & (margins < 0)).sum())
    return {"n": len(records), "accepted": n, "rejected": len(records) - n,
            "coverage": n / len(records), "errors": errors,
            "risk": errors / n if n else None, "accuracy": 1 - errors / n if n else None,
            "retained_false_negatives": false_negatives,
            "retained_fn_fraction_of_all_ad": false_negatives / int((labels == 1).sum())
            if (labels == 1).any() else None}


def risk_coverage(records):
    """Describe TEST ranking with whole score ties and common coverage targets.

    These thresholds are descriptive curve coordinates, not deployed policies.
    AURC uses right-endpoint steps over complete tied groups.
    """
    records = list(records)
    scores = np.abs([row["margin"] for row in records])
    errors = np.asarray([(row["margin"] >= 0) != row["label"] for row in records])
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(records)]
    cumulative = np.cumsum(errors[order])
    points = [{"coverage": int(end) / len(records), "accepted": int(end),
               "risk": int(cumulative[end - 1]) / int(end),
               "score_threshold": float(scores[order[end - 1]])} for end in ends]
    area, previous = 0.0, 0.0
    for point in points:
        area += (point["coverage"] - previous) * point["risk"]
        previous = point["coverage"]
    common = [{"target_coverage": target / 10, **next(
        point for point in points if point["accepted"] * 10 >= target * len(records))}
              for target in range(1, 11)]
    return {"points": points, "common_coverage": common, "aurc": area}


def paired_bootstrap(results, replicates=2000, seed=9001):
    """Paired diagnosis-stratified patient resampling of slice metrics.

    Every sampled patient contributes all slices, with multiplicity. Shared
    draws compare both models/seeds; fitted models and calibration stay fixed.
    Undefined replicates invalidate an interval rather than being dropped.
    """
    results = sorted(results, key=lambda item: (item["model"], item["seed"]))
    populations = [sorted(item["records"], key=lambda row: row["path"]) for item in results]
    reference = [(row["path"], row["subject_id"], row["label"]) for row in populations[0]]
    if any([(r["path"], r["subject_id"], r["label"]) for r in rows] != reference
           for rows in populations):
        raise ValueError("Paired intervals require identical slice/subject/label joins")
    keys = [(item["model"], item["seed"]) for item in results]
    seeds = sorted({key[1] for key in keys})
    if len(set(keys)) != len(keys) or set(keys) != {(m, s) for m in ("cnn", "gfnet") for s in seeds}:
        raise ValueError("Need matching CNN/GFNet seeds without duplicates")
    labels = np.asarray([row[2] for row in reference])
    patients, group = np.unique([row[1] for row in reference], return_inverse=True)
    diagnosis = np.asarray([labels[group == i][0] for i in range(len(patients))])
    if not np.array_equal(labels, diagnosis[group]):
        raise ValueError("Patient diagnosis must be consistent for stratification")
    weights = np.zeros((replicates + 1, len(patients)))
    weights[0] = 1
    rng = np.random.Generator(np.random.PCG64(seed))
    for label in (0, 1):
        members = np.flatnonzero(diagnosis == label)
        draws = rng.choice(members, size=(replicates, len(members)), replace=True)
        np.add.at(weights, (np.arange(1, replicates + 1)[:, None], draws), 1)

    def total(values):
        patient_total = np.bincount(group, weights=np.asarray(values, dtype=float), minlength=len(patients))
        return np.einsum("ij,j->i", weights, patient_total, optimize=False)

    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)

    count, ad_count = total(np.ones(len(labels))), total(labels == 1)
    series = {}
    for item, rows, key in zip(results, populations, keys):
        z = np.asarray([row["margin"] for row in rows])
        wrong = (z >= 0) != labels
        scaled = z / item["policy"]["temperature"]
        keep = np.abs(scaled) >= item["policy"]["score_threshold"]
        metrics = {"accuracy": total(~wrong) / count,
                   "coverage": total(keep) / count,
                   "selective_risk": ratio(total(keep & wrong), total(keep)),
                   "retained_fn_fraction_of_all_ad": ratio(total(keep & (labels == 1) & (z < 0)), ad_count)}
        for prefix, margins in (("raw", z), ("calibrated", scaled)):
            probability = probabilities(margins)
            confidence = probabilities(np.abs(margins))
            high = confidence >= .9
            metrics[prefix + "_nll"] = total(np.logaddexp(0, margins) - labels * margins) / count
            metrics[prefix + "_brier"] = total((probability - labels) ** 2) / count
            metrics[prefix + "_hc_error_rate"] = ratio(total(high & wrong), total(high))
            deviation = np.zeros(replicates + 1)
            edges = np.linspace(.5, 1, 11)
            for lower, upper in zip(edges[:-1], edges[1:]):
                inside = (confidence >= lower) & ((confidence <= upper) if upper == 1 else (confidence < upper))
                deviation += np.abs(total(inside & ~wrong) - total(inside * confidence))
            metrics[prefix + "_ece"] = deviation / count
        series[key] = metrics

    def summarize(metrics):
        output = {}
        for name, values in metrics.items():
            valid = int(np.isfinite(values[1:]).sum())
            output[name] = {"point": float(values[0]) if np.isfinite(values[0]) else None,
                            "ci95": np.percentile(values[1:], [2.5, 97.5]).tolist() if valid == replicates else None,
                            "valid_replicates": valid}
        return output

    means = {model: {name: np.mean([series[model, s][name] for s in seeds], axis=0)
                     for name in next(iter(series.values()))} for model in ("cnn", "gfnet")}
    differences = {str(s): summarize({name: series["gfnet", s][name] - series["cnn", s][name]
                                     for name in means["cnn"]}) for s in seeds}
    differences["mean"] = summarize({name: means["gfnet"][name] - means["cnn"][name]
                                     for name in means["cnn"]})
    return {"replicates": replicates, "seed": seed, "patients": len(patients), "slices": len(labels),
            "sampling": "paired diagnosis-stratified patient clusters; all slice multiplicities retained",
            "per_seed": {f"{m}-{s}": summarize(value) for (m, s), value in series.items()},
            "means": {model: summarize(value) for model, value in means.items()},
            "gfnet_minus_cnn": differences}


def failure_cases(results):
    """Five fixed seed-3710 error slots, with separate predictions for all seeds."""
    views = {(item["model"], item["seed"]): {r["path"]: r for r in item["records"]} for item in results}
    policies = {(item["model"], item["seed"]): item["policy"] for item in results}
    cnn, gfnet = views["cnn", 3710], views["gfnet", 3710]
    paths = sorted(cnn)
    wrong = lambda row: (row["margin"] >= 0) != row["label"]
    slots = [
        ("GFNet confident false negative", lambda p: gfnet[p]["label"] == 1 and gfnet[p]["margin"] <= -math.log(9), lambda p: -abs(gfnet[p]["margin"])),
        ("CNN confident false negative", lambda p: cnn[p]["label"] == 1 and cnn[p]["margin"] <= -math.log(9), lambda p: -abs(cnn[p]["margin"])),
        ("GFNet-only error", lambda p: wrong(gfnet[p]) and not wrong(cnn[p]), lambda p: -abs(gfnet[p]["margin"])),
        ("CNN-only error", lambda p: wrong(cnn[p]) and not wrong(gfnet[p]), lambda p: -abs(cnn[p]["margin"])),
        ("GFNet wrong near boundary", lambda p: wrong(gfnet[p]), lambda p: abs(gfnet[p]["margin"])),
    ]
    selected, used = [], set()
    for name, eligible, rank in slots:
        available = [p for p in paths if gfnet[p]["subject_id"] not in used]
        candidates = sorted((p for p in available if eligible(p)), key=lambda p: (rank(p), p))
        fallback = not candidates
        if fallback:
            candidates = sorted((p for p in available if wrong(gfnet[p]) or wrong(cnn[p])),
                                key=lambda p: (-abs(gfnet[p]["margin"]), p))
        if not candidates:
            continue
        path = candidates[0]
        used.add(gfnet[path]["subject_id"])
        predictions = {}
        for key, records in views.items():
            row = records[path]
            predictions[f"{key[0]}-{key[1]}"] = {
                "prediction": int(row["margin"] >= 0), "margin": row["margin"],
                "raw_confidence": float(probabilities(abs(row["margin"]))),
                "calibrated_confidence": float(probabilities(abs(row["margin"]) / policies[key]["temperature"]))}
        selected.append({"slot": name, "used_fallback": fallback, "path": path,
                         "subject_id": gfnet[path]["subject_id"], "label": gfnet[path]["label"],
                         "predictions": predictions})
    return selected


def plot_reliability(report, output, title="Slice confidence reliability"):
    """Plot existing bin statistics and slice counts without rerunning inference."""
    import matplotlib.pyplot as plt

    bins = report["reliability_bins"]
    populated = [entry for entry in bins if entry["count"]]
    fig, (ax, counts_ax) = plt.subplots(
        2, 1, figsize=(6.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained",
    )
    ax.plot([0.5, 1], [0.5, 1], "--", color="0.5", label="Ideal calibration")
    ax.plot([entry["confidence"] for entry in populated],
            [entry["accuracy"] for entry in populated],
            "o-", color="#2563EB", label="Observed accuracy")
    ax.set(title=f"{title}\nECE = {report['ece']:.3f} | n = {report['n']:,}",
           ylabel="Fraction correct", ylim=(0, 1.03), xlim=(0.49, 1.01))
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    counts_ax.bar([(entry["lower"] + entry["upper"]) / 2 for entry in bins],
                  [entry["count"] for entry in bins],
                  width=[0.9 * (entry["upper"] - entry["lower"]) for entry in bins],
                  color="#2563EB")
    counts_ax.set(xlabel="Predicted confidence", ylabel="Slices")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def preparation_sensitivity(results, manifest_rows):
    """Describe source slice-start/rank strata without changing the main cohort."""
    lookup = {row["relative_image_path"]: row for row in manifest_rows if row["partition"] == "test"}
    if not all("slice_index" in row and "slice_rank" in row for row in lookup.values()):
        return {"status": "Source slice-index/rank metadata unavailable"}
    groups = {}
    for path, row in lookup.items():
        start = int(row["slice_index"]) - int(row["slice_rank"])
        for name in (f"source_start_{start}", f"rank_quarter_{int(row['slice_rank']) // 5 + 1}"):
            groups.setdefault(name, set()).add(path)
    output = {}
    for name, paths in sorted(groups.items()):
        strata = []
        for item in results:
            rows = [row for row in item["records"] if row["path"] in paths]
            scaled = scale_confidence(rows, item["policy"]["temperature"])
            strata.append({"model": item["model"], "seed": item["seed"],
                           "subjects": len({r["subject_id"] for r in rows}),
                           "raw": reliability_report(rows), "calibrated": reliability_report(scaled),
                           "selective": selective_report(scaled, item["policy"]["score_threshold"])})
        output[name] = strata
    return output


def plot_final_results(results, cases, output, data_root=None):
    """Plot saved metrics/predictions; optional case images need no model pass."""
    import matplotlib.pyplot as plt

    output = Path(output)
    results = sorted(results, key=lambda item: (item["model"], item["seed"]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, item in zip(axes.flat, results):
        ax.plot([.5, 1], [.5, 1], "--", color=".6")
        for kind, color in (("raw", "#2563EB"), ("calibrated", "#D97706")):
            bins = [b for b in item[kind]["reliability_bins"] if b["count"]]
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color=color,
                    label=f"{kind}: ECE {item[kind]['ece']:.3f}")
        ax.set(title=f"{item['model'].upper()} / {item['seed']}", xlim=(.49, 1.01), ylim=(0, 1.02),
               xlabel="Mean confidence", ylabel="Fraction correct")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output / "reliability.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for item in results:
        points = item["risk_coverage"]["points"]
        color = "#2563EB" if item["model"] == "cnn" else "#D97706"
        ax.step([0] + [p["coverage"] for p in points],
                [points[0]["risk"]] + [p["risk"] for p in points], where="pre",
                color=color, alpha=.65, label=f"{item['model'].upper()} / {item['seed']}")
        if item["selective"]["risk"] is not None:
            ax.scatter(item["selective"]["coverage"], item["selective"]["risk"], color=color, s=35)
    ax.set(xlabel="Coverage (fraction of TEST slices retained)", ylabel="Selective error rate",
           title="Slice risk--coverage; dots use CALIBRATION-only policies", xlim=(0, 1), ylim=(0, 1))
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.2)
    fig.savefig(output / "risk_coverage.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    for ax, item in zip(axes.flat, results):
        matrix = np.asarray(item["raw"]["confusion"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(sum(matrix)))
        for i in range(2):
            for j in range(2):ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NC", "AD"], yticklabels=["NC", "AD"],
               xlabel="Predicted", ylabel="True", title=f"{item['model'].upper()} {item['seed']}: {item['raw']['accuracy']:.2%}")
    fig.savefig(output / "confusion.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, field, title, scale in zip(axes, ("parameters", "peak_training_allocated_bytes", "forward_ms_per_slice"),
                                      ("Parameters (millions)", "Recorded training peak (GiB)", "Forward throughput (ms/slice)"),
                                      (1e6, 2 ** 30, 1)):
        for index, model in enumerate(("cnn", "gfnet")):
            items = [r for r in results if r["model"] == model]
            values = [(r[field] if field == "parameters" else r.get("training_cost", {}).get(field)
                       if field.startswith("peak_training") else r["profile"][field]) for r in items]
            values = [v / scale for v in values if v is not None]
            ax.bar(index, np.mean(values), color=("#2563EB", "#D97706")[index], alpha=.6)
            ax.scatter(np.full(len(values), index), values, color="black", s=18)
        ax.set(xticks=[0, 1], xticklabels=["CNN", "GFNet"], title=title); ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "cost.png", dpi=180); plt.close(fig)

    if data_root is not None and cases:
        from PIL import Image
        fig, axes = plt.subplots(len(cases), 2, figsize=(12, 2.7 * len(cases)), squeeze=False, layout="constrained")
        for (image_ax, text_ax), case in zip(axes, cases):
            with Image.open(Path(data_root) / case["path"]) as image:image_ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
            image_ax.axis("off"); text_ax.axis("off")
            lines = [case["slot"] + (" (fallback)" if case["used_fallback"] else ""),
                     f"True {'AD' if case['label'] else 'NC'} | {case['subject_id']}", case["path"]]
            for key, value in sorted(case["predictions"].items()):
                lines.append(f"{key}: {'AD' if value['prediction'] else 'NC'}; raw {value['raw_confidence']:.3f}, calibrated {value['calibrated_confidence']:.3f}")
            text_ax.text(0, .98, "\n".join(lines), va="top", fontsize=8.5)
        fig.savefig(output / "failure_cases.png", dpi=160); plt.close(fig)


def analyze_evaluations(inputs, manifest, output, data_root=None):
    """Compose one saved-record analysis, paired intervals, strata and figures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a fresh final-analysis directory")
    results = [json.loads(Path(path).read_text()) for path in inputs]
    rows = read_manifest(manifest)
    expected = sorted((r["relative_image_path"], r["subject_id"], r["label"]) for r in rows if r["partition"] == "test")
    if {(r["model"], r["seed"]) for r in results} != {(m, s) for m in ("cnn", "gfnet") for s in (3710, 3711, 3712)} or len(results) != 6:
        raise ValueError("Final analysis requires the six frozen model/seed outputs")
    for item in results:
        actual = sorted((r["path"], r["subject_id"], r["label"]) for r in item["records"])
        if item["partition"] != "test" or item["manifest_sha256"] != file_digest(manifest) or actual != expected:
            raise ValueError("Final results must match every declared TEST slice")
    cases = failure_cases(results)
    result = {"bootstrap": paired_bootstrap(results), "failure_cases": cases,
              "preparation_sensitivity": preparation_sensitivity(results, rows),
              "models": [{k: v for k, v in item.items() if k not in ("records", "risk_coverage")} for item in results],
              "input_hashes": {str(path): file_digest(path) for path in inputs}}
    write_json(output / "analysis.json", result)
    plot_final_results(results, cases, output, data_root)
    lines = ["# Final individual-slice evaluation", "", "All seeds share the same TEST patients; these are not independent cohorts.", "",
             "| Model | Seed | Correct / slices | Accuracy | Raw / calibrated NLL | Raw / calibrated ECE | Fixed coverage / risk | HC errors / FNs (raw) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in sorted(results, key=lambda r: (r["model"], r["seed"])):
        a, b, s = item["raw"], item["calibrated"], item["selective"]
        risk = f"{s['risk']:.2%}" if s["risk"] is not None else "undefined"
        lines.append(f"| {item['model']} | {item['seed']} | {a['correct']} / {a['n']} | {a['accuracy']:.2%} | {a['nll']:.4f} / {b['nll']:.4f} | {a['ece']:.4f} / {b['ece']:.4f} | {s['coverage']:.2%} / {risk} | {a['high_confidence_errors']} / {a['high_confidence_false_negatives']} |")
    lines += ["", "analysis.json contains paired conditional intervals, all classification/uncertainty metrics, source-start/rank sensitivity, error cases and cost measurements.",
              "Thresholds and temperatures were fixed from CALIBRATION; accuracy includes every TEST slice. Compare selective risk alongside achieved coverage.",
              "Source-start/rank strata are descriptive and cannot establish removal of preparation confounding or causal disease features.",
              "The study compares selected model/training recipes, not an isolated Fourier-operator ablation. Prior cohort exposure and repeated TUNE development limit generalization claims.",
              "Clinical diagnostic ambiguity and external-site validity were not established. The engineering recommendation must weigh reliability/errors before accuracy and cost."]
    (output / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    return result


def infer(checkpoint, manifest, data_root, partition, batch_size=64, *,
          device="cpu", output=None):
    """Predict TUNE/CALIBRATION slices; TEST requires its later fixed-policy path."""
    if partition not in ("tune", "calibration"):
        raise ValueError("Use tune or calibration; final TEST evaluation is not enabled here")
    rows = read_manifest(manifest)
    model, metadata = load_selected_model(checkpoint, device)
    manifest_sha256 = file_digest(manifest)
    if metadata["identity"]["manifest_sha256"] != manifest_sha256:
        raise ValueError("Checkpoint and prediction manifest must match")
    config = metadata["identity"]["config"]
    data = ADNISlices(rows, data_root, partition,
                      input_scale=config.get("input_scale", "zero_one"))
    result = {
        "format": "slice-predictions-v1", "partition": partition,
        "model": config["model"], "seed": config["seed"],
        "checkpoint_epoch": metadata["epoch"],
        "checkpoint_sha256": file_digest(checkpoint), "manifest_sha256": manifest_sha256,
        "records": predict_slices(model, make_loader(data, batch_size, config["seed"]), device),
    }
    result["metrics"] = reliability_report(result["records"])
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    return result


def evaluate_selected(checkpoint, manifest, data_root, output, batch_size=64, *,
                      device="cpu", runtime=None):
    """Fit CALIBRATION policy, then evaluate every TEST slice exactly once.

    The selected best weights stay fixed in one evaluation model. Save the
    calibration stage before TEST; this fresh-run command never overwrites a
    previous evaluation. It creates no optimizer or saved-training RNG restore.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a fresh final-evaluation output directory")
    rows = read_manifest(manifest)
    manifest_sha256 = file_digest(manifest)
    checkpoint_sha256 = file_digest(checkpoint)
    model, metadata = load_selected_model(checkpoint, device)
    if metadata["identity"]["manifest_sha256"] != manifest_sha256:
        raise ValueError("Checkpoint and final-evaluation manifest must match")
    config = metadata["identity"]["config"]
    identity = {"model": config["model"], "seed": config["seed"],
                "checkpoint_epoch": metadata["epoch"], "checkpoint_sha256": checkpoint_sha256,
                "manifest_sha256": manifest_sha256,
                "training_identity": metadata["identity"], "runtime": runtime or {},
                "evaluation_sources": {name: file_digest(Path(__file__).with_name(name))
                                       for name in ("dataset.py", "modules.py", "train.py", "predict.py")}}

    def predict_partition(partition):
        data = ADNISlices(rows, data_root, partition,
                          input_scale=config.get("input_scale", "zero_one"))
        profile = {}
        records = predict_slices(model, make_loader(data, batch_size, config["seed"]),
                                 device, profile=profile)
        return records, profile

    calibration, calibration_profile = predict_partition("calibration")
    policy = fit_calibration(calibration)
    scaled = scale_confidence(calibration, policy["temperature"])
    write_json(output / "calibration.json", {
        **identity, "partition": "calibration", "records": calibration, "policy": policy,
        "raw": reliability_report(calibration), "calibrated": reliability_report(scaled),
        "profile": calibration_profile,
    })
    print(f"calibration_complete model={config['model']} seed={config['seed']} n={len(calibration)}", flush=True)
    records, profile = predict_partition("test")
    scaled = scale_confidence(records, policy["temperature"])
    result = {**identity, "format": "final-slice-evaluation-v1", "partition": "test",
              "records": records, "policy": policy,
              "raw": reliability_report(records), "calibrated": reliability_report(scaled),
              "selective": selective_report(scaled, policy["score_threshold"]),
              "risk_coverage": risk_coverage(records), "profile": profile,
              "parameters": sum(parameter.numel() for parameter in model.parameters())}
    history_path = Path(checkpoint).with_name("history.json")
    if history_path.exists():
        history = json.loads(history_path.read_text())
        result["training_cost"] = {
            "epochs": len(history),
            "train_epoch_seconds": [item["train"]["elapsed_s"] for item in history],
            "tune_epoch_seconds": [item["tune"]["elapsed_s"] for item in history],
            "peak_training_allocated_bytes": max(item["train"]["peak_allocated_bytes"] or 0 for item in history),
            "peak_training_reserved_bytes": max(item["train"]["peak_reserved_bytes"] or 0 for item in history),
        }
    write_json(output / "test.json", result)
    print(f"test_complete model={config['model']} seed={config['seed']} n={len(records)}", flush=True)
    return result


def main(argv=None):
    """Run allocated-GPU prediction or report existing records on CPU."""
    parser = argparse.ArgumentParser(description="ADNI slice prediction and reliability reporting")
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict", help="Predict TUNE/CALIBRATION on an allocated A100")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--manifest", required=True)
    prediction.add_argument("--data-root", required=True)
    prediction.add_argument("--partition", choices=("tune", "calibration"), required=True)
    prediction.add_argument("--batch-size", type=int, default=64)
    prediction.add_argument("--seed", type=int, default=3710,
                            help="Runtime initialization seed; trained model identity comes from checkpoint")
    prediction.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate", help="One CALIBRATION/TEST pass for frozen best weights")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--data-root", required=True)
    evaluation.add_argument("--output", required=True, help="Fresh directory for calibration.json and test.json")
    evaluation.add_argument("--batch-size", type=int, default=64)
    evaluation.add_argument("--seed", type=int, default=3710)
    reporting = commands.add_parser("report", help="Analyze saved predictions without rerunning the model")
    reporting.add_argument("--input", required=True)
    reporting.add_argument("--output", required=True)
    reporting.add_argument("--bins", type=int, default=10)
    reporting.add_argument("--figure", help="Optional reliability diagram PNG/PDF")
    analysis = commands.add_parser("analyze", help="Analyze six saved final evaluations without model execution")
    analysis.add_argument("--inputs", nargs=6, required=True)
    analysis.add_argument("--manifest", required=True)
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--data-root", help="Optional original image root for five case illustrations")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        result = analyze_evaluations(args.inputs, args.manifest, args.output, args.data_root)
        print(json.dumps({"patients": result["bootstrap"]["patients"], "slices": result["bootstrap"]["slices"],
                          "replicates": result["bootstrap"]["replicates"]}, indent=2))
        return 0
    if args.command in ("predict", "evaluate"):
        device, runtime = a100_runtime(args.seed)
        if args.command == "evaluate":
            result = evaluate_selected(args.checkpoint, args.manifest, args.data_root, args.output,
                                       args.batch_size, device=device, runtime=runtime)
            print(json.dumps({"model": result["model"], "seed": result["seed"],
                              "test_accuracy": result["raw"]["accuracy"],
                              "temperature": result["policy"]["temperature"]}, indent=2))
            return 0
        result = infer(args.checkpoint, args.manifest, args.data_root, args.partition,
                       args.batch_size, device=device)
        result["runtime"] = runtime
    else:
        result = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result["metrics"] = reliability_report(result["records"], args.bins)
        if args.figure:
            plot_reliability(result["metrics"], args.figure)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    metrics = result["metrics"]
    print(json.dumps({key: metrics[key] for key in ("n", "correct", "accuracy", "nll", "brier", "ece")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
