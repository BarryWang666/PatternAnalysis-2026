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


def load_selected_model(checkpoint, device="cpu"):
    """Load best.pt on device and return (evaluation model, checkpoint metadata).

    The configuration chooses CNN/GFNet; training optimizer and RNG are not
    restored. latest.pt is reserved for resume and is rejected here.
    """
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "sampled-best-v1":
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


def reliability_report(records, n_bins=5):
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
    data = ADNISlices(rows, data_root, partition)
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


def hierarchy_report(rows, records):
    """Report raw slices, conventional scans, patients and weighted scans."""
    grouped=scan_summary(rows,records)
    scans=grouped['scans']
    patients=defaultdict(list)
    for row in scans: patients[row['subject_id']].append(row)
    subjects=[dict(label=v[0]['label'],probability_ad=math.fsum(r['probability_ad'] for r in v)/len(v))
              for _,v in sorted(patients.items())]
    def summarize(values):
        return weighted_probability_metrics([r['label'] for r in values],[r['probability_ad'] for r in values])
    return dict(slice=reliability_report(records),scan=summarize(scans),subject=summarize(subjects),
                subject_weighted_scan=grouped['weighted_scan_metrics'])


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


def paired_subject_bootstrap(rows, predictions, policies, replicates=2000, seed=9001):
    """Shared diagnosis-stratified patient draws, preserving visits and multiplicity.

    predictions/policies map (model, training_seed) to saved records/fixed policy.
    No model training or calibration fitting occurs. Percentile intervals are
    conditional on the fitted models, policies, cohort and patient partition.
    """
    groups=defaultdict(list)
    for row in rows: groups[row['subject_id']].append(row)
    strata=[[s for s in sorted(groups) if int(groups[s][0]['label'])==label] for label in (0,1)]
    if any(not ids for ids in strata): raise ValueError('Both diagnosis strata are required')
    expected={(m,s) for m in ('cnn','gfnet') for s in (3710,3711,3712)}
    if set(predictions)!=expected or set(policies)!=expected: raise ValueError('Need the fixed six-model roster')
    tables={slot:{r['path']:r for r in records} for slot,records in predictions.items()}
    def point(selected,slot):
        samples=[];joined=[]
        for draw_id,subject in enumerate(selected):
            for row in groups[subject]:
                path=f"{draw_id}:{row['relative_image_path']}"
                samples.append(dict(row,relative_image_path=path,subject_id=f'{draw_id}:{subject}'))
                joined.append(dict(tables[slot][row['relative_image_path']],path=path,subject_id=f'{draw_id}:{subject}'))
        raw=classification_report(joined)
        calibrated=apply_scan_policy(policies[slot],samples,joined)
        return dict(slice_accuracy=raw['accuracy'],slice_nll=raw['nll'],
                    selective_risk=calibrated['selective']['risk'])
    def flatten(selected):
        per={slot:point(selected,slot) for slot in sorted(expected)}
        values={f'{m}/{s}/{metric}':v for (m,s),metrics in per.items() for metric,v in metrics.items()}
        for metric in next(iter(per.values())):
            means={m:float(np.mean([per[m,s][metric] for s in (3710,3711,3712)]))
                   if all(per[m,s][metric] is not None for s in (3710,3711,3712)) else None for m in ('cnn','gfnet')}
            values.update({f'{m}/mean/{metric}':v for m,v in means.items()})
            values[f'gfnet-minus-cnn/mean/{metric}']=means['gfnet']-means['cnn'] if all(v is not None for v in means.values()) else None
            for s in (3710,3711,3712):
                a,b=per['gfnet',s][metric],per['cnn',s][metric]
                values[f'gfnet-minus-cnn/{s}/{metric}']=a-b if a is not None and b is not None else None
        return values
    points=flatten(sorted(groups)); draws={k:[] for k in points}
    rng=np.random.Generator(np.random.PCG64(seed))
    for _ in range(replicates):
        selected=[ids[int(i)] for ids in strata for i in rng.integers(len(ids),size=len(ids))]
        for key,value in flatten(selected).items(): draws[key].append(value)
    return {key:dict(point=value,interval=np.percentile(draws[key],[2.5,97.5],method='linear').tolist()
                    if all(v is not None for v in draws[key]) else None,
                    valid_replicates=sum(v is not None for v in draws[key]),replicates=replicates)
            for key,value in points.items()}


def select_failure_cases(rows, predictions, policies):
    """Five seed-3710 error slots, immediate fallback, distinct patients.

    Raw predictions define errors. Calibrated confidence ranks error cases;
    both-wrong fallback ranks within-scan GFNet slice-probability SD (ddof=0).
    """
    tables={m:scan_summary(rows,recs)['scans'] for m,recs in predictions.items()}
    paths={m:{r['path']:float(probabilities(r['margin'])) for r in recs} for m,recs in predictions.items()}
    members=defaultdict(list)
    for row in rows: members[row['subject_id'],row['scan_id']].append(row['relative_image_path'])
    cases=[]
    for i,gf in enumerate(tables['gfnet']):
        cnn=tables['cnn'][i]
        p,c,y=gf['probability_ad'],cnn['probability_ad'],gf['label']
        q=float(temperature_scale([p],policies['gfnet']['temperature'])[0])
        cq=float(temperature_scale([c],policies['cnn']['temperature'])[0])
        sd=float(np.std([paths['gfnet'][path] for path in members[gf['subject_id'],gf['scan_id']]],ddof=0))
        cases.append(dict(subject_id=gf['subject_id'],scan_id=gf['scan_id'],label=y,
            gfnet_probability=p,cnn_probability=c,gfnet_confidence=max(q,1-q),cnn_confidence=max(cq,1-cq),
            gfnet_error=(p>=.5)!=y,cnn_error=(c>=.5)!=y,gfnet_slice_sd=sd))
    slots=[('gfnet_false_negative',lambda c:c['label']==1 and c['gfnet_error']),
           ('gfnet_false_positive',lambda c:c['label']==0 and c['gfnet_error']),
           ('gfnet_nearest_boundary_error',lambda c:c['gfnet_error']),
           ('gfnet_wrong_cnn_correct',lambda c:c['gfnet_error'] and not c['cnn_error']),
           ('cnn_wrong_gfnet_correct',lambda c:c['cnn_error'] and not c['gfnet_error'])]
    result=[];used=set()
    for name,matches in slots:
        available=[c for c in cases if c['subject_id'] not in used]
        candidates=[c for c in available if matches(c)]
        fallback=not candidates
        both_wrong=False
        if fallback:
            errors=[c for c in available if c['gfnet_error'] or c['cnn_error']]
            candidates=[c for c in errors if c['gfnet_error'] and c['cnn_error']]
            both_wrong=bool(candidates)
            if not candidates: candidates=errors
        def rank(c):
            if fallback:
                score=-c['gfnet_slice_sd'] if both_wrong else -c['gfnet_confidence' if c['gfnet_error'] else 'cnn_confidence']
            elif name=='gfnet_nearest_boundary_error': score=abs(c['gfnet_probability']-.5)
            elif name=='cnn_wrong_gfnet_correct': score=-c['cnn_confidence']
            else: score=-c['gfnet_confidence']
            return score,c['subject_id'],c['scan_id']
        chosen=min(candidates,key=rank) if candidates else None
        if chosen: used.add(chosen['subject_id'])
        result.append(dict(slot=name,fallback=fallback,case=chosen))
    return result


def preparation_sensitivity(rows,records,policy):
    """Describe acquisition slice-start strata with unchanged fitted policy."""
    starts=defaultdict(list)
    for row in rows: starts[(row['subject_id'],row['scan_id'])].append(int(row['slice_index']))
    groups={key:min(indices) for key,indices in starts.items()}
    reports={}
    for name,allowed in [('full',set(groups.values())),('common_78_88',{78,88})]+[(f'start_{s}',{s}) for s in sorted(set(groups.values()))]:
        included=[r for r in rows if groups[r['subject_id'],r['scan_id']] in allowed]
        paths={r['relative_image_path'] for r in included}
        predictions=[r for r in records if r['path'] in paths]
        reports[name]=dict(slice_count=len(included),subject_count=len({r['subject_id'] for r in included}),
            raw=hierarchy_report(included,predictions) if included else None,
            calibrated=apply_scan_policy(policy,included,predictions) if included else None)
    return reports


def infer_released_test(checkpoint,manifest,data_root,release,model_name,seed,device='cpu'):
    """Use an explicit six-model release roster and previously fixed scan policy."""
    slot=f'{model_name}-{seed}'
    entry=release['models'][slot]
    if len(release['models'])!=6 or file_digest(checkpoint)!=entry['checkpoint_sha256']:
        raise ValueError('Selected checkpoint does not match the six-model release')
    if file_digest(manifest)!=release['manifest_sha256']:
        raise ValueError('Release population changed')
    model,metadata=load_selected_model(checkpoint,device)
    config=metadata['identity']['config']
    if config['model']!=model_name or config['seed']!=seed:
        raise ValueError('Model/seed differs from released slot')
    rows=read_manifest(manifest);data=ADNISlices(rows,data_root,'test')
    records=predict_slices(model,make_loader(data,64,seed),device)
    return dict(model=model_name,seed=seed,partition='test',records=records,
                raw=hierarchy_report(data.rows,records),
                calibrated=apply_scan_policy(entry['policy'],data.rows,records))


if __name__ == '__main__':
    main()
