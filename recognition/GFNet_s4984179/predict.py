"""COMP3710 | s4984179 | Individual slice inference and reliability analysis.

Input: selected weights and image metadata, or saved prediction records.
Output: per-slice probabilities, classification/confidence metrics and plots.
Metrics operate on slices; patient IDs are metadata rather than predictions.
"""

import argparse
import json
from pathlib import Path
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
    model = build_model(saved["identity"]["config"]["model"])
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    metadata = {key: value for key, value in saved.items() if key != "model"}
    return model, metadata


def predict_slices(model, loader, device="cpu"):
    """Return one record per loader image, including the final partial batch.

    Records contain path, subject ID, label, class, margin, AD probability and
    confidence. Evaluation/inference mode prevents parameter or gradient updates.
    """
    model.to(device).eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device))
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
    reporting = commands.add_parser("report", help="Analyze saved predictions without rerunning the model")
    reporting.add_argument("--input", required=True)
    reporting.add_argument("--output", required=True)
    reporting.add_argument("--bins", type=int, default=10)
    reporting.add_argument("--figure", help="Optional reliability diagram PNG/PDF")
    args = parser.parse_args(argv)
    if args.command == "predict":
        device, runtime = a100_runtime(args.seed)
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
