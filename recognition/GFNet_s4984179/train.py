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


from modules import BaselineCNN, GFNet


@dataclass(frozen=True)
class Config:
    model: str
    lr: float
    seed: int = 3710
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 60
    warmup_epochs: int = 5
    min_epochs: int = 30
    patience: int = 12
    min_delta: float = 1e-4

    def validate(self):
        if self.model not in ("cnn", "gfnet"):
            raise ValueError("Choose cnn or gfnet")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if (not 0 <= self.seed < 2**32 or self.batch_size < 1 or self.patience < 1
                or not 0 <= self.warmup_epochs < self.max_epochs
                or not 1 <= self.min_epochs <= self.max_epochs):
            raise ValueError("Invalid seed, batch size or epoch limits")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def a100_runtime(seed):
    """Initialize the assigned A100, deterministic FP32, seed and 2/1 threads.

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
    torch.set_num_interop_threads(1)
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


def build_model(name):
    return {"cnn": BaselineCNN, "gfnet": GFNet}[name]()


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
        fraction = step / max(1, warmup_steps - 1)
    else:
        fraction = (1 + math.cos(math.pi * (step - warmup_steps)
                                / max(1, total_steps - warmup_steps - 1))) / 2
    return peak * (0.01 + 0.99 * fraction)


def is_better(candidate, best):
    """Strictly minimize subject-weighted scan NLL; exact ties stay earlier."""
    return best is None or candidate["weighted_scan_nll"] < best["weighted_scan_nll"]


def advance_selection(tune, best, bad_epochs, epoch, config, anchor=None):
    """Keep strict-best weights separately from the min-delta patience anchor."""
    value = tune["weighted_scan_nll"]
    if anchor is None or value <= anchor - config.min_delta:
        anchor, bad_epochs = value, 0
    else:
        bad_epochs += 1
    selected = is_better(tune,best)
    if selected:
        best = {"epoch":epoch,"n":tune["n"],"correct":tune["correct"],
                "nll":tune["nll"],"weighted_scan_nll":value}
    reason = "epoch_cap" if epoch == config.max_epochs else (
        "patience" if epoch >= config.min_epochs and bad_epochs >= config.patience else None)
    return best,bad_epochs,selected,reason,anchor


def probabilities(margins):
    """Stable AD probabilities from AD-minus-NC logits."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(margins, dtype=np.float64)))


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


def run_epoch(model, loader, device, *, optimizer=None, step=0, config=None):
    """Train with an optimizer, otherwise evaluate without gradients.

    Consume all sampler slices and return (metrics, per-slice records, step).
    Training metrics are online; evaluation uses fixed weights and sample-
    weighted loss. When config is provided, step drives the global LR schedule.
    """
    training = optimizer is not None
    model.train(training)
    records, loss_sum, clipped = [], 0.0, 0
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
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                clipped += int(norm > 1)
                optimizer.step()
                step += 1
            loss_sum += loss.item() * len(labels)
            margins = (logits[:, 1] - logits[:, 0]).detach().cpu().tolist()
            records.extend(
                {"path": path, "subject_id": subject, "label": int(label), "margin": margin}
                for path, subject, label, margin in zip(
                    batch["path"], batch["subject_id"], batch["label"].tolist(), margins,
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not records or len(records) != len(loader.sampler):
        raise ValueError("Epoch must cover the declared sampler")
    metrics = slice_metrics([row["label"] for row in records], [row["margin"] for row in records])
    if not training:
        metrics["weighted_scan_nll"] = scan_summary(loader.dataset.rows, records)["weighted_scan_nll"]
    metrics.update({
        "mean_loss": loss_sum / len(records), "elapsed_s": time.perf_counter() - started,
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
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def fit(model, train_data, tune_data, config, output, manifest_sha256, device,
        runtime, *, resume=False, epochs=None):
    """Fit TRAIN/TUNE datasets and persist complete epoch boundaries.

    epochs limits this invocation; config.max_epochs retains the LR horizon.
    resume restores latest weights, optimizer, RNG, progress and patience.
    Return epoch history, selecting best by subject-weighted scan NLL.
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
        if (saved["format"] != "sampled-latest-v1"
                or saved["identity"]["config"] != asdict(config)
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
    identity = {"config": asdict(config), "manifest_sha256": manifest_sha256,
                "sources": source_identity()}
    runtime = {**runtime, "sources": source_identity()}
    history, best, best_checkpoint = [], None, None
    bad_epochs, step, completed = 0, 0, 0
    anchor = None
    if saved is not None:
        identity = saved["identity"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_checkpoint = saved["history"], saved["best"], saved["best_checkpoint"]
        bad_epochs, step, completed = saved["bad_epochs"], saved["step"], saved["epoch"]
        anchor = saved["anchor"]
        # latest is authoritative if a stop interrupted the separate best export.
        save_checkpoint(output / "best.pt", best_checkpoint)
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
        tune_result, _, _ = run_epoch(model, tune_loader, device)
        best, bad_epochs, selected, reason, anchor = advance_selection(
            tune_result, best, bad_epochs, epoch, config, anchor,
        )
        if selected:
            best_checkpoint = {
                "format": "sampled-best-v1", "identity": identity, "epoch": epoch,
                "tune": tune_result,
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        history.append({"epoch": epoch, "updates": step, "train": train_result,
                        "tune": tune_result, "lr": optimizer.param_groups[0]["lr"], "runtime": runtime})
        status = "complete" if reason else "paused"
        save_checkpoint(output / "latest.pt", {
            "format": "sampled-latest-v1", "identity": identity, "epoch": epoch,
            "step": step, "best": best, "bad_epochs": bad_epochs, "anchor": anchor,
            "status": status, "stop_reason": reason,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "history": history,
            "rng": capture_rng(device), "best_checkpoint": best_checkpoint,
        })
        if selected:
            save_checkpoint(output / "best.pt", best_checkpoint)
        write_json(output / "history.json", history)
        write_json(output / "result.json", {
            "status": status, "epoch": epoch, "updates": step,
            "stop_reason": reason, "best": best,
            "train_presentations": epoch * len(train_loader.sampler),
            "tune_presentations": epoch * len(tune_data),
        })
        print(f"epoch={epoch} lr={optimizer.param_groups[0]['lr']:.6g} "
              f"train_acc={train_result['accuracy']:.4f} tune_acc={tune_result['accuracy']:.4f} "
              f"tune_nll={tune_result['nll']:.4f} "
              f"hc_fn={tune_result['high_confidence_false_negatives']}", flush=True)
        if reason:
            break
    return history


def experiment_plan():
    """Reserve six seed-3710 LR fits and four additional selected-LR fits."""
    peaks=(1e-4,3e-4,1e-3)
    development=[dict(model=m,seed=3710,lr=lr) for m in ("cnn","gfnet") for lr in peaks]
    additional=[dict(model=m,seed=seed,lr=None) for m in ("cnn","gfnet") for seed in (3711,3712)]
    return dict(development=development,additional=additional,reuse_winning_seed=3710)


def select_development(records):
    """Require all six candidates, then minimize best weighted scan NLL."""
    records=list(records)
    expected={(m,3710,lr) for m in ("cnn","gfnet") for lr in (1e-4,3e-4,1e-3)}
    keys=[(r["model"],r["seed"],r["lr"]) for r in records]
    if len(keys)!=6 or set(keys)!=expected or any(r["status"]!="complete" for r in records):
        raise ValueError("Selection requires exactly six completed development fits")
    return {m:min((r for r in records if r["model"]==m),
        key=lambda r:(r["best"]["weighted_scan_nll"],(1e-4,3e-4,1e-3).index(r["lr"])))
        for m in ("cnn","gfnet")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("cnn", "gfnet"), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, help="Additional complete epochs this invocation; keep the global LR horizon")
    parser.add_argument("--resume", action="store_true", help="Continue this output directory's latest.pt")
    args = parser.parse_args()
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    config.validate()
    source, output = Path(__file__).parent.resolve(), Path(args.output).resolve()
    if output == source or source in output.parents:
        parser.error("Keep run outputs outside the source folder")
    rows = read_manifest(args.manifest)
    train_data = ADNISlices(rows, args.data_root, "train")
    tune_data = ADNISlices(rows, args.data_root, "tune")
    device, runtime = a100_runtime(config.seed)
    print(json.dumps({"train_slices": len(train_data), "tune_slices": len(tune_data),
                      "config": asdict(config), "runtime": runtime}), flush=True)
    fit(build_model(config.model), train_data, tune_data, config, output,
        file_digest(args.manifest), device, runtime, resume=args.resume, epochs=args.epochs)


if __name__ == "__main__":
    main()
