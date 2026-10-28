"""COMP3710 | s4984179 | Equal-slice training and TUNE selection.

Input: slice manifest, image root and CNN/GFNet configuration.
Output: selected-best/latest weights, epoch metrics and a run summary.
Usage: python train.py --help; production runs require an allocated A100.
"""

import argparse
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
from modules import BaselineCNN, GFNet, GFNetPyramid


@dataclass(frozen=True)
class Config:
    model: str
    lr: float
    seed: int = 3710
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 30
    warmup_epochs: int = 2
    min_epochs: int = 20
    patience: int = 8
    input_scale: str = "zero_one"
    augment: bool = False
    gfnet_embedding: str = "patch"
    label_smoothing: float = 0.0
    gfnet_depth: int = 12
    pyramid_depths: tuple = (2, 2, 2)
    mixup_alpha: float = 0.0
    gfnet_mlp_ratio: int = 4

    def validate(self):
        if self.model not in ("cnn", "gfnet", "gfnet_pyramid"):
            raise ValueError("Choose cnn, gfnet or gfnet_pyramid")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if self.input_scale not in ("zero_one", "minus_one_one"):
            raise ValueError("Choose zero_one or minus_one_one input scaling")
        if self.gfnet_embedding not in ("patch", "conv") or (
                self.model != "gfnet" and self.gfnet_embedding != "patch"):
            raise ValueError("Embedding choice applies to the flat GFNet")
        if not math.isfinite(self.label_smoothing) or not 0 <= self.label_smoothing < 1:
            raise ValueError("Label smoothing must be in [0,1)")
        if not math.isfinite(self.mixup_alpha) or self.mixup_alpha < 0:
            raise ValueError("MixUp alpha must be finite and nonnegative")
        if (not isinstance(self.gfnet_depth, int) or self.gfnet_depth < 1
                or (self.model != "gfnet" and self.gfnet_depth != 12)):
            raise ValueError("Choose a positive flat-GFNet depth; other models use its default")
        if not isinstance(self.gfnet_mlp_ratio, int) or self.gfnet_mlp_ratio not in (2, 4) or (
                self.model != "gfnet" and self.gfnet_mlp_ratio != 4):
            raise ValueError("MLP ratio 2 or 4 applies to flat GFNet; other models use 4")
        if (len(self.pyramid_depths) != 3
                or any(not isinstance(depth, int) or depth < 1 for depth in self.pyramid_depths)
                or (self.model != "gfnet_pyramid" and tuple(self.pyramid_depths) != (2, 2, 2))):
            raise ValueError("Choose three positive Pyramid depths; other models use its default")
        if (not 0 <= self.seed < 2**32 or self.batch_size < 1 or self.patience < 1
                or not 0 <= self.warmup_epochs < self.max_epochs
                or not 1 <= self.min_epochs <= self.max_epochs):
            raise ValueError("Invalid seed, batch size or epoch limits")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def a100_runtime(seed):
    """Initialize the assigned A100, deterministic FP32, seed and 2/2 threads.

    Return a CUDA device and measured runtime metadata. Call once at entry,
    before model construction or computation; keep Slurm's device assignment.
    """
    partition = os.environ.get("SLURM_JOB_PARTITION", "").split(",")[0]
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit()
            or os.environ.get("SLURM_JOB_ACCOUNT") != "comp3710"
            or partition not in ("a100", "a100-grind", "a100-test", "comp3710")):
        raise RuntimeError("Run inside a comp3710 allocation on an A100 partition")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One allocated CUDA device is required")
    hardware = torch.cuda.get_device_properties(0)
    if "A100" not in hardware.name:
        raise RuntimeError(f"Expected an A100, found {hardware.name}")
    seed_everything(seed)
    runtime = {
        "job": os.environ["SLURM_JOB_ID"], "node": platform.node(),
        "partition": partition, "gpu": hardware.name,
        "uuid": str(getattr(hardware, "uuid", "unavailable")),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "numpy": np.__version__,
        "threads": [torch.get_num_threads(), torch.get_num_interop_threads()],
    }
    return torch.device("cuda:0"), runtime


def build_model(name, gfnet_embedding="patch", gfnet_depth=12, pyramid_depths=(2, 2, 2),
                gfnet_mlp_ratio=4):
    """Return a fresh CNN, GFNet or pyramid model with the requested structure."""
    return {"cnn": BaselineCNN,
            "gfnet": lambda: GFNet(embedding=gfnet_embedding, depth=gfnet_depth,
                                   mlp_ratio=gfnet_mlp_ratio),
            "gfnet_pyramid": lambda: GFNetPyramid(stage_depths=pyramid_depths)}[name]()


def build_optimizer(model, config):
    """AdamW decays weight matrices, excluding biases, norms and positions."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name == "pos_embed" else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr, betas=(0.9, 0.999), foreach=False,
    )


def learning_rate(step, total_steps, warmup_steps, peak):
    """Warm up to peak, then cosine-decay to 1%; use a global update cursor."""
    if not 0 <= step < total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError("Learning-rate cursor is outside its declared horizon")
    if step < warmup_steps:
        fraction = (step + 1) / warmup_steps
    else:
        fraction = (1 + math.cos(math.pi * (step - warmup_steps)
                                / max(1, total_steps - warmup_steps - 1))) / 2
    return peak * (0.01 + 0.99 * fraction)


def is_better(candidate, best):
    """More correct TUNE slices, then lower NLL; equal ties keep the earlier fit."""
    if best is not None and candidate["n"] != best["n"]:
        raise ValueError("Compare the same complete TUNE population")
    return best is None or (candidate["correct"], -candidate["nll"]) > (best["correct"], -best["nll"])


def advance_selection(tune, best, bad_epochs, epoch, config):
    """An NLL-only tie-break can select weights but never reset accuracy patience."""
    improved = best is None or tune["correct"] > best["correct"]
    bad_epochs = 0 if improved else bad_epochs + 1
    selected = is_better(tune, best)
    if selected:
        best = {"epoch": epoch, "n": tune["n"], "correct": tune["correct"], "nll": tune["nll"]}
    reason = None
    if epoch == config.max_epochs:
        reason = "epoch_cap"
    elif epoch >= config.min_epochs and bad_epochs >= config.patience:
        reason = "patience"
    return best, bad_epochs, selected, reason


def probabilities(margins):
    """Stable AD probabilities from AD-minus-NC logits."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(margins, dtype=np.float64)))


def mixup_batch(images, labels, alpha):
    """Interpolate a TRAIN batch and return its two label vectors and weight.

    One Beta(alpha, alpha) draw and one permutation are shared by the batch.
    NumPy/Torch RNG states are already captured by epoch-boundary checkpoints.
    Alpha zero preserves images and RNG exactly, retaining the baseline path.
    """
    if alpha == 0:
        return images, labels, labels, 1.0
    weight = float(np.random.beta(alpha, alpha))
    order = torch.randperm(len(images), device=images.device)
    return (weight * images + (1 - weight) * images[order],
            labels, labels[order], weight)


def slice_metrics(labels, margins):
    """Compute equal-slice metrics from aligned 1D labels and logit margins.

    Labels are NC=0/AD=1; margin is AD logit minus NC logit. Return accuracy,
    NLL, class recalls and confidence>=0.90 mistakes over every supplied slice.
    """
    y, z = np.asarray(labels), np.asarray(margins, dtype=np.float64)
    if (y.ndim != 1 or z.shape != y.shape or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(z).all()):
        raise ValueError("Need aligned binary labels and finite slice margins")
    predicted = z >= 0
    correct = predicted == y
    confidence = np.maximum(probabilities(z), probabilities(-z))
    matrix = [[int(np.sum((y == truth) & (predicted == decision)))
               for decision in (0, 1)] for truth in (0, 1)]
    high = confidence >= 0.9
    return {
        "n": len(y), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
        "nll": float(np.mean(np.logaddexp(0.0, z) - y * z)),
        "confusion": matrix,
        "recall": {name: matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None
                   for i, name in enumerate(("NC", "AD"))},
        "high_confidence_count": int(high.sum()),
        "high_confidence_errors": int((high & ~correct).sum()),
        "high_confidence_false_negatives": int((high & (y == 1) & ~predicted).sum()),
        "mean_confidence_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
    }


def run_epoch(model, loader, device, *, optimizer=None, step=0, config=None):
    """Train with an optimizer, otherwise evaluate without gradients.

    Consume all loader slices and return (metrics, per-slice records, step).
    Training metrics are online; evaluation uses fixed weights and sample-
    weighted loss. When config is provided, step drives the global LR schedule.
    """
    training = optimizer is not None
    mixed = training and config is not None and config.mixup_alpha > 0
    model.train(training)
    records, loss_sum, clipped, consumed = [], 0.0, 0, 0
    initial_step = step
    free_bytes = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        free_bytes = torch.cuda.mem_get_info(device)[0]
    started = time.perf_counter()
    with (torch.enable_grad() if training else torch.inference_mode()):
        for batch in loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            if training:
                if config is not None:
                    lr = learning_rate(step, config.max_epochs * len(loader),
                                       config.warmup_epochs * len(loader), config.lr)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
            if mixed:
                images, first_labels, second_labels, weight = mixup_batch(
                    images, labels, config.mixup_alpha,
                )
            logits = model(images)
            smoothing = config.label_smoothing if training and config is not None else 0.0
            if mixed:
                loss = (weight * F.cross_entropy(logits, first_labels, label_smoothing=smoothing)
                        + (1 - weight) * F.cross_entropy(
                            logits, second_labels, label_smoothing=smoothing))
            else:
                loss = F.cross_entropy(logits, labels, label_smoothing=smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                clipped += int(norm > 1)
                optimizer.step()
                step += 1
            loss_sum += loss.item() * len(labels)
            consumed += len(labels)
            if mixed:
                continue  # Mixed images have no single genuine class or slice prediction.
            margins = (logits[:, 1] - logits[:, 0]).detach().cpu().tolist()
            records.extend(
                {"path": path, "subject_id": subject, "label": int(label), "margin": margin}
                for path, subject, label, margin in zip(
                    batch["path"], batch["subject_id"], batch["label"].tolist(), margins,
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not consumed or consumed != len(loader.dataset):
        raise ValueError("Epoch must cover every slice")
    metrics = ({"n": consumed, "objective": "mixup", "accuracy": None, "nll": None}
               if mixed else slice_metrics(
                   [row["label"] for row in records], [row["margin"] for row in records]))
    metrics.update({
        "mean_loss": loss_sum / consumed, "elapsed_s": time.perf_counter() - started,
        "updates": step - initial_step, "clipped_updates": clipped,
        "initially_free_bytes": free_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    })
    return metrics, records, step


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    return {name: file_digest(Path(__file__).with_name(name))
            for name in ("dataset.py", "modules.py", "train.py")}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng(device):
    """Return Python, NumPy, CPU Torch and active-device CUDA RNG states."""
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    """Restore capture_rng's saved states before continuing training on device."""
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_tune_predictions(output, selected):
    """Export the selected epoch's existing unmixed TUNE records, if available.

    Keeping them inside the selected payload lets resume repair this export
    without another model pass. Historical weights may have no such records.
    """
    if "tune_records" in selected:
        write_json(Path(output) / "best-tune.json", {
            "format": "slice-tune-predictions-v1", "partition": "tune",
            "identity": selected["identity"], "epoch": selected["epoch"],
            "metrics": selected["tune"], "records": selected["tune_records"],
        })


def plot_history(history, output):
    """Draw saved TRAIN objectives and TUNE metrics without model execution."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 3.5), layout="constrained")
    loss_ax, accuracy_ax = figure.subplots(1, 2)
    epochs = [item["epoch"] for item in history]
    loss_ax.plot(epochs, [item["train"]["mean_loss"] for item in history], label="TRAIN objective")
    loss_ax.plot(epochs, [item["tune"]["nll"] for item in history], label="TUNE NLL")
    for role in ("train", "tune"):
        values = [item[role]["accuracy"] for item in history]
        if any(value is not None for value in values):
            accuracy_ax.plot(epochs, values, label=role.upper())
    loss_ax.set(xlabel="Epoch", ylabel="Loss")
    accuracy_ax.set(xlabel="Epoch", ylabel="Slice accuracy", ylim=(0, 1))
    for axis in (loss_ax, accuracy_ax):
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(output, dpi=180)


def fit(model, train_data, tune_data, config, output, manifest_sha256, device,
        runtime, *, resume=False, epochs=None):
    """Fit TRAIN/TUNE datasets and persist complete epoch boundaries.

    epochs limits this invocation; config.max_epochs retains the LR horizon.
    resume restores latest weights, optimizer, RNG, progress and patience.
    Return epoch history, selecting best by TUNE correct count then NLL.
    """
    config.validate()
    if train_data.partition != "train" or tune_data.partition != "tune":
        raise ValueError("Fit accepts only TRAIN and TUNE")
    if epochs is not None and epochs < 1:
        raise ValueError("Segment length must be positive")
    output = Path(output)
    saved = None
    if resume:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        previous_config = {"mixup_alpha": 0.0, "gfnet_mlp_ratio": 4,
                           **saved["identity"]["config"]}
        if (saved["format"] != "slice-latest-v1"
                or previous_config != asdict(config)
                or saved["identity"]["manifest_sha256"] != manifest_sha256):
            raise ValueError("Resume requires matching configuration and manifest")
        if saved["status"] == "complete":
            raise ValueError("This fit already completed")
        if (saved["rng"]["cuda"] is None) != (device.type == "cpu"):
            raise ValueError("Resume on the same device type")
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError("Use an empty output directory or --resume")
    model.to(device)
    optimizer = build_optimizer(model, config)
    sources = source_identity()
    identity = {"config": asdict(config), "manifest_sha256": manifest_sha256,
                "sources": sources}
    runtime = {**runtime, "sources": sources}
    history, best, best_checkpoint = [], None, None
    bad_epochs, step, completed = 0, 0, 0
    if saved is not None:
        identity = saved["identity"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_checkpoint = saved["history"], saved["best"], saved["best_checkpoint"]
        bad_epochs, step, completed = saved["bad_epochs"], saved["step"], saved["epoch"]
        # latest is authoritative if a stop interrupted the separate best export.
        save_checkpoint(output / "best.pt", best_checkpoint)
        save_tune_predictions(output, best_checkpoint)
        restore_rng(saved["rng"], device)
    else:
        write_json(output / "config.json", {**identity, "runtime": runtime})
    stop_epoch = min(config.max_epochs, completed + (epochs or config.max_epochs))
    for epoch in range(completed + 1, stop_epoch + 1):
        train_loader = make_loader(train_data, config.batch_size, config.seed, epoch - 1)
        tune_loader = make_loader(tune_data, config.batch_size, config.seed)
        train_result, _, step = run_epoch(
            model, train_loader, device, optimizer=optimizer, step=step, config=config,
        )
        tune_result, tune_records, _ = run_epoch(model, tune_loader, device)
        best, bad_epochs, selected, reason = advance_selection(
            tune_result, best, bad_epochs, epoch, config,
        )
        if selected:
            best_checkpoint = {
                "format": "slice-best-v1", "identity": identity, "epoch": epoch,
                "tune": tune_result, "tune_records": tune_records,
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        history.append({"epoch": epoch, "updates": step, "train": train_result,
                        "tune": tune_result, "lr": optimizer.param_groups[0]["lr"], "runtime": runtime})
        status = "complete" if reason else "paused"
        save_checkpoint(output / "latest.pt", {
            "format": "slice-latest-v1", "identity": identity, "epoch": epoch,
            "step": step, "best": best, "bad_epochs": bad_epochs,
            "status": status, "stop_reason": reason,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "history": history,
            "rng": capture_rng(device), "best_checkpoint": best_checkpoint,
        })
        if selected:
            save_checkpoint(output / "best.pt", best_checkpoint)
            save_tune_predictions(output, best_checkpoint)
        write_json(output / "history.json", history)
        write_json(output / "result.json", {
            "status": status, "epoch": epoch, "updates": step,
            "stop_reason": reason, "best": best,
            "train_presentations": epoch * len(train_data),
            "tune_presentations": epoch * len(tune_data),
        })
        train_summary = (f"train_mixup_loss={train_result['mean_loss']:.4f}"
                         if config.mixup_alpha > 0 else f"train_acc={train_result['accuracy']:.4f}")
        print(f"epoch={epoch} lr={optimizer.param_groups[0]['lr']:.6g} "
              f"{train_summary} tune_acc={tune_result['accuracy']:.4f} "
              f"tune_nll={tune_result['nll']:.4f} "
              f"hc_fn={tune_result['high_confidence_false_negatives']}", flush=True)
        if reason:
            break
    plot_history(history, output / "learning_curves.png")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("cnn", "gfnet", "gfnet_pyramid"), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--input-scale", choices=("zero_one", "minus_one_one"), default="zero_one")
    parser.add_argument("--augment", action="store_true", help="TRAIN-only rotation +/-5 degrees and translation +/-2%%")
    parser.add_argument("--gfnet-embedding", choices=("patch", "conv"), default="patch")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="TRAIN cross-entropy smoothing; TUNE NLL stays unsmoothed")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="TRAIN-only batch MixUp Beta parameter; zero disables mixing")
    parser.add_argument("--gfnet-depth", type=int, default=12)
    parser.add_argument("--gfnet-mlp-ratio", type=int, choices=(2, 4), default=4,
                        help="Flat GFNet channel expansion; historical checkpoints use 4")
    parser.add_argument("--pyramid-depths", type=int, nargs=3, default=(2, 2, 2),
                        metavar=("FINE", "MIDDLE", "COARSE"),
                        help="Fourier block counts at 60x64, 30x32 and 15x16")
    parser.add_argument("--epochs", type=int, help="Additional complete epochs this invocation; keep the global LR horizon")
    parser.add_argument("--resume", action="store_true", help="Continue this output directory's latest.pt")
    args = parser.parse_args()
    values = {key: getattr(args, key) for key in Config.__dataclass_fields__}
    values["pyramid_depths"] = tuple(values["pyramid_depths"])
    config = Config(**values)
    config.validate()
    source, output = Path(__file__).parent.resolve(), Path(args.output).resolve()
    if output == source or source in output.parents:
        parser.error("Keep run outputs outside the source folder")
    rows = read_manifest(args.manifest)
    train_data = ADNISlices(rows, args.data_root, "train", input_scale=config.input_scale,
                            augment=config.augment)
    tune_data = ADNISlices(rows, args.data_root, "tune", input_scale=config.input_scale)
    device, runtime = a100_runtime(config.seed)
    print(json.dumps({"train_slices": len(train_data), "tune_slices": len(tune_data),
                      "config": asdict(config), "runtime": runtime}), flush=True)
    fit(build_model(config.model, config.gfnet_embedding, config.gfnet_depth,
                    config.pyramid_depths, config.gfnet_mlp_ratio),
        train_data, tune_data, config, output,
        file_digest(args.manifest), device, runtime, resume=args.resume, epochs=args.epochs)


if __name__ == "__main__":
    main()
